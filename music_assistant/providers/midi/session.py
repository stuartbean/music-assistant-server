"""Bounded, sample-addressed sidecar audio driven solely by MA's stereo consumer."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, Self, TypeVar

from .constants import BLOCK_FRAMES, MAX_CONSUMERS, PCM_CHANNELS, RETAINED_FRAMES, SAMPLE_RATE
from .routing import FAMILIES
from .synthesizer import MidiRenderer, mix_stereo

if TYPE_CHECKING:
    from .midi_file import MidiFile

_T = TypeVar("_T")
SessionState = Literal["pending", "rendering", "eof", "failed", "closed"]


class SessionClosedError(RuntimeError):
    """The producer was closed before it reached normal end-of-stream."""


class FramesUnavailableError(RuntimeError):
    """Requested frames were discarded or the consumer exceeded the retention window."""


@dataclass(frozen=True)
class PCMFormat:
    """Internal interleaved PCM format with seven named stereo instrument-family buses."""

    sample_rate: int = SAMPLE_RATE
    representation: str = "IEEE-754 float32"
    byte_order: str = "little"
    channel_layout: tuple[str, ...] = tuple(
        f"{family}_{side}" for family in FAMILIES for side in ("L", "R")
    )
    family_names: tuple[str, ...] = tuple(family.value for family in FAMILIES)
    bytes_per_frame: int = PCM_CHANNELS * 4


@dataclass(frozen=True)
class PCMBlock:
    """A contiguous interval on the original MIDI file's sample timeline."""

    start_frame: int
    frame_count: int
    data: bytes


class MidiConsumer:
    """An attached sequential reader that never advances synthesis."""

    def __init__(
        self, session: MidiSession, start_frame: int, *, follow_origin: bool = False
    ) -> None:
        """Create a reader positioned on its session timeline."""
        self.session = session
        self.next_frame = start_frame
        self._follow_origin = follow_origin
        self._error: Exception | None = None
        self._detached = False
        self._reading = False

    async def read(self, frame_count: int = BLOCK_FRAMES) -> PCMBlock | None:
        """
        Wait for up to frame_count retained frames, returning None only at normal EOF.

        The final block can be short. Cancellation leaves the cursor unchanged.
        Concurrent reads on the same consumer are rejected.

        :param frame_count: Requested interval length, bounded by the retention capacity.
        :raises FramesUnavailableError: If this consumer fell behind the bounded window.
        :raises SessionClosedError: On explicit detach, stop, cancellation or provider unload.
        """
        if not 0 < frame_count <= self.session.capacity_frames:
            raise ValueError("Read size must be positive and fit in the retention window")
        if self._reading:
            raise RuntimeError("A MIDI consumer supports only one outstanding read")
        self._reading = True
        try:
            return await self.session.read(self, frame_count)
        finally:
            self._reading = False

    def detach(self) -> None:
        """Detach immediately, wake pending reads and release this consumer's retained audio."""
        self.session.detach(self)

    async def __aenter__(self) -> Self:
        """Return this attached consumer."""
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Detach on context exit."""
        self.detach()


class MidiSession:
    """One renderer, one file timeline and a bounded optional multichannel history."""

    format = PCMFormat()

    def __init__(
        self,
        session_id: str,
        midi: MidiFile,
        soundfont: str,
        first_frame: int = 0,
        *,
        capacity_frames: int = RETAINED_FRAMES,
        on_idle: Callable[[], None] = lambda: None,
    ) -> None:
        """Reserve a render session without creating a synth or retaining audio."""
        self.session_id = session_id
        self.first_frame = first_frame
        self.total_frames = midi.total_frames
        self.capacity_frames = capacity_frames
        self.state: SessionState = "pending"
        self.error: Exception | None = None
        self.end_frame = first_frame
        self._midi = midi
        self._soundfont = soundfont
        self._on_idle = on_idle
        self._blocks: deque[PCMBlock] = deque()
        self._consumers: set[MidiConsumer] = set()
        self._changed = asyncio.Event()
        self._executor: ThreadPoolExecutor | None = None
        self._renderer: MidiRenderer | None = None
        self._cleanup_task: asyncio.Task[None] | None = None

    def is_rendering(self) -> bool:
        """Return whether standard audio consumption currently owns a live render."""
        return self.state == "rendering"

    @property
    def retained_frames(self) -> int:
        """Return the count of retained multichannel frames (zero without consumers)."""
        return sum(block.frame_count for block in self._blocks)

    @property
    def earliest_frame(self) -> int:
        """Return the oldest frame available for attachment, or the current write head."""
        return self._blocks[0].start_frame if self._blocks else self.end_frame

    def attach(self, start_frame: int | None = None) -> MidiConsumer:
        """
        Attach at the current write head, or at an explicitly retained frame.

        Attach before stereo consumption to receive the whole piece. Attaching does
        not instantiate or advance the synth. No audio is retained without consumers.

        :param start_frame: Absolute file frame; None subscribes from the write head.
        """
        if self.state in ("closed", "failed"):
            raise SessionClosedError(f"MIDI session is {self.state}") from self.error
        frame = self.end_frame if start_frame is None else start_frame
        if not self.earliest_frame <= frame <= self.end_frame:
            raise FramesUnavailableError(
                f"Frame {frame} is unavailable; retained interval is "
                f"[{self.earliest_frame}, {self.end_frame})"
            )
        if len(self._consumers) >= MAX_CONSUMERS:
            raise RuntimeError("Too many multichannel consumers for this MIDI session")
        consumer = MidiConsumer(self, frame, follow_origin=start_frame is None)
        self._consumers.add(consumer)
        return consumer

    def set_source_origin(self, first_frame: int) -> None:
        """
        Bind a pending session to the actual provider-generator seek origin.

        MA may fill its stereo buffer from before the requested playback position.
        Explicit consumer cursors keep their file positions; default cursors follow
        the actual source origin until the render starts.

        :param first_frame: First file frame the standard provider generator will yield.
        """
        if self.state != "pending":
            raise RuntimeError("A live MIDI rendering session cannot change source origin")
        if not 0 <= first_frame <= self.total_frames:
            raise ValueError("MIDI source origin is outside the performance")
        self.first_frame = self.end_frame = first_frame
        for consumer in self._consumers:
            if consumer._follow_origin:
                consumer.next_frame = first_frame

    def detach(self, consumer: MidiConsumer) -> None:
        """Detach a consumer without affecting ordinary stereo playback."""
        consumer._detached = True
        self._consumers.discard(consumer)
        if not self._consumers:
            self._blocks.clear()
            if self.state in ("eof", "failed", "closed"):
                self._on_idle()
        self._changed.set()

    async def read(self, consumer: MidiConsumer, frames: int) -> PCMBlock | None:
        """Read a consumer's next retained interval; this never calls the synthesizer."""
        while True:
            if consumer._error:
                raise consumer._error
            if consumer._detached or self.state == "closed":
                raise SessionClosedError("MIDI consumer or session was closed")
            if self.state == "failed":
                raise SessionClosedError("MIDI rendering failed") from self.error
            if consumer.next_frame < self.earliest_frame:
                raise FramesUnavailableError("The consumer fell behind the retained MIDI audio")
            available = self.end_frame - consumer.next_frame
            if available >= frames or self.state == "eof":
                count = min(frames, available)
                if count == 0:
                    return None
                start = consumer.next_frame
                end = start + count
                parts = []
                for block in self._blocks:
                    left = max(start, block.start_frame)
                    right = min(end, block.start_frame + block.frame_count)
                    if left < right:
                        parts.append(
                            block.data[
                                (left - block.start_frame) * self.format.bytes_per_frame : (
                                    right - block.start_frame
                                )
                                * self.format.bytes_per_frame
                            ]
                        )
                consumer.next_frame = end
                return PCMBlock(start, count, b"".join(parts))
            self._changed.clear()
            await self._changed.wait()

    async def stereo_stream(self) -> AsyncGenerator[bytes]:
        """Drive the sole synthesis pass, yielding stereo derived from each published block."""
        if self.state != "pending":
            raise RuntimeError("A MIDI session can be rendered only once")
        self.state = "rendering"
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="midi-render")
        try:
            await self._run(self._create_renderer)
            frame = 0
            while frame < self.first_frame and self.is_rendering():
                count = min(BLOCK_FRAMES, self.first_frame - frame)
                await self._run(partial(self._render, count))
                frame += count
            while self.end_frame < self.total_frames and self.is_rendering():
                stems, stereo = await self._run(partial(self._render, BLOCK_FRAMES))
                if not self.is_rendering():
                    break
                self._publish(stems)
                yield stereo
            if self.is_rendering():
                self.state = "eof"
        except Exception as err:
            self.error = err
            self.state = "failed"
            raise
        finally:
            if self.is_rendering():
                self.state = "closed"
            if self.state in ("closed", "failed"):
                self._blocks.clear()
                self._consumers.clear()
            self._changed.set()
            await self._cleanup()
            if not self._consumers:
                self._on_idle()

    async def close(self) -> None:
        """Close the session, wake readers and release native resources, including during a render."""
        self.state = "closed"
        self._blocks.clear()
        self._consumers.clear()
        self._changed.set()
        await self._cleanup()
        self._on_idle()

    async def _run(self, function: Callable[[], _T]) -> _T:
        # Shield keeps native work queued/running until the same worker can close
        # it. A cancelled await must never race deletion of a live synth pointer.
        return await asyncio.shield(
            asyncio.get_running_loop().run_in_executor(self._executor, function)
        )

    def _create_renderer(self) -> None:
        self._renderer = MidiRenderer(self._midi, self._soundfont)

    def _render(self, frames: int) -> tuple[bytes, bytes]:
        assert self._renderer is not None
        pcm = self._renderer.render(frames)
        return pcm.astype("<f4", copy=False).tobytes(), mix_stereo(pcm)

    def _publish(self, data: bytes) -> None:
        count = len(data) // self.format.bytes_per_frame
        block = PCMBlock(self.end_frame, count, data)
        self.end_frame += count
        cutoff = max(self.first_frame, self.end_frame - self.capacity_frames)
        for consumer in tuple(self._consumers):
            if consumer.next_frame < cutoff:
                consumer._error = FramesUnavailableError(
                    f"MIDI consumer overrun: frame {consumer.next_frame} predates {cutoff}"
                )
                self.detach(consumer)
        if self._consumers:
            self._blocks.append(block)
            while (
                self._blocks and self._blocks[0].start_frame + self._blocks[0].frame_count <= cutoff
            ):
                self._blocks.popleft()
            if self._blocks and self._blocks[0].start_frame < cutoff:
                first = self._blocks.popleft()
                skip = cutoff - first.start_frame
                self._blocks.appendleft(
                    PCMBlock(
                        cutoff,
                        first.frame_count - skip,
                        first.data[skip * self.format.bytes_per_frame :],
                    )
                )
        self._changed.set()

    async def _cleanup(self) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._close_renderer())
        await asyncio.shield(self._cleanup_task)

    async def _close_renderer(self) -> None:
        executor = self._executor
        if executor is None:
            return
        try:
            await asyncio.get_running_loop().run_in_executor(executor, self._delete_renderer)
        finally:
            executor.shutdown(wait=False)

    def _delete_renderer(self) -> None:
        if self._renderer:
            self._renderer.close()
            self._renderer = None
