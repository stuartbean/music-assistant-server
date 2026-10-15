"""Single-render, bounded-buffer and resource-lifecycle contract tests."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar
from unittest.mock import patch

import numpy as np
import pytest

from music_assistant.providers.midi.constants import BLOCK_FRAMES, PCM_CHANNELS
from music_assistant.providers.midi.midi_file import MidiFile, load_midi
from music_assistant.providers.midi.session import (
    FramesUnavailableError,
    MidiSession,
    SessionClosedError,
)
from music_assistant.providers.midi.synthesizer import mix_stereo

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray


class FakeRenderer:
    """A deterministic renderer that records calls and thread ownership."""

    instances: ClassVar[list[FakeRenderer]] = []

    def __init__(self, midi: MidiFile, _soundfont: str) -> None:
        """Record the sole renderer instance and thread."""
        self.frame = 0
        self.total = midi.total_frames
        self.calls = 0
        self.closed = False
        self.thread = threading.get_ident()
        self.instances.append(self)

    def render(self, frames: int) -> NDArray[np.float32]:
        """Emit sample-addressable PCM independent of caller chunking."""
        assert self.thread == threading.get_ident()
        self.calls += 1
        count = min(frames, self.total - self.frame)
        values = np.arange(self.frame, self.frame + count, dtype=np.float32) / np.float32(1000000)
        result = np.repeat(values[:, None], PCM_CHANNELS, axis=1)
        self.frame += count
        return result

    def close(self) -> None:
        """Record deletion on the original rendering thread."""
        assert self.thread == threading.get_ident()
        self.closed = True


@pytest.fixture(autouse=True)
def fake_renderer() -> Generator[None]:
    """Replace only the native renderer, leaving session scheduling and buffering real."""
    FakeRenderer.instances.clear()
    with patch("music_assistant.providers.midi.session.MidiRenderer", FakeRenderer):
        yield


def make_session(frames: int = BLOCK_FRAMES * 4, first_frame: int = 0) -> MidiSession:
    """Construct a session with a small synthetic timeline."""
    midi = MidiFile("Test", (), frames, frames)
    return MidiSession("test", midi, "unused.sf2", first_frame, capacity_frames=BLOCK_FRAMES * 2)


async def test_attachment_never_renders_and_mix_matches() -> None:
    """The stereo consumer advances exactly once and sidecar reads reuse those exact frames."""
    session = make_session()
    consumer = session.attach(0)
    assert not FakeRenderer.instances
    source = session.stereo_stream()
    waiting = asyncio.create_task(consumer.read(1000))
    await asyncio.sleep(0)
    assert not waiting.done()
    stereo = await anext(source)
    block = await waiting
    assert block is not None
    assert (block.start_frame, block.frame_count) == (0, 1000)
    pcm = np.frombuffer(block.data, dtype="<f4").reshape(-1, PCM_CHANNELS)
    assert mix_stereo(pcm) == stereo[: 1000 * 8]
    assert len(FakeRenderer.instances) == 1
    assert FakeRenderer.instances[0].calls == 1
    remainder = await consumer.read(BLOCK_FRAMES - 1000)
    assert remainder is not None
    assert remainder.start_frame == 1000
    assert FakeRenderer.instances[0].calls == 1
    await source.aclose()
    assert FakeRenderer.instances[0].closed
    assert session.retained_frames == 0
    with pytest.raises(SessionClosedError):
        await consumer.read()


async def test_eof_and_zero_retention_without_subscribers() -> None:
    """Ordinary playback keeps no multichannel history and releases its synth at EOF."""
    session = make_session(BLOCK_FRAMES + 7)
    stereo = [chunk async for chunk in session.stereo_stream()]
    assert sum(map(len, stereo)) == (BLOCK_FRAMES + 7) * 8
    assert session.state == "eof"
    assert session.retained_frames == 0
    assert FakeRenderer.instances[0].closed
    with pytest.raises(FramesUnavailableError):
        session.attach(0)
    consumer = session.attach()
    assert await consumer.read() is None
    consumer.detach()


async def test_slow_consumer_fails_without_stalling_stereo() -> None:
    """A stale consumer is dropped explicitly while the full stereo stream completes."""
    session = make_session()
    slow = session.attach(0)
    async for _chunk in session.stereo_stream():
        assert session.retained_frames <= session.capacity_frames
    with pytest.raises(FramesUnavailableError, match="overrun"):
        await slow.read()
    assert session.state == "eof"
    assert session.retained_frames == 0


async def test_independent_readers_detach_and_short_final_block() -> None:
    """Readers have separate cursors and one detaching does not stop another."""
    session = make_session(BLOCK_FRAMES + 17)
    first = session.attach()
    second = session.attach()
    source = session.stereo_stream()
    await anext(source)
    first.detach()
    with pytest.raises(SessionClosedError):
        await first.read()
    assert (block := await second.read()) is not None
    assert block.start_frame == 0
    await anext(source)
    with pytest.raises(StopAsyncIteration):
        await anext(source)
    final = await second.read()
    assert final is not None
    assert final.frame_count == 17
    assert await second.read() is None
    second.detach()
    assert session.retained_frames == 0


async def test_cancelled_read_does_not_advance_cursor() -> None:
    """Cancelling a blocked sidecar read neither renders nor consumes frames."""
    session = make_session()
    consumer = session.attach()
    waiting = asyncio.create_task(consumer.read())
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert consumer.next_frame == 0
    assert not FakeRenderer.instances
    await session.close()


async def test_close_wakes_waiting_reader() -> None:
    """Provider unload can close even a session whose stereo generator never started."""
    session = make_session()
    consumer = session.attach()
    waiting = asyncio.create_task(consumer.read())
    await asyncio.sleep(0)
    await session.close()
    with pytest.raises(SessionClosedError):
        await waiting
    assert not FakeRenderer.instances


async def test_seek_discards_frames_using_the_same_renderer() -> None:
    """Seeking renders all prior state but publishes only frames at/after the seek origin."""
    origin = BLOCK_FRAMES + 123
    session = make_session(first_frame=origin)
    consumer = session.attach(origin)
    source = session.stereo_stream()
    stereo = await anext(source)
    block = await consumer.read()
    assert block is not None
    assert block.start_frame == origin
    pcm = np.frombuffer(block.data, dtype="<f4").reshape(-1, PCM_CHANNELS)
    assert pcm[0, 0] == np.float32(origin / np.float32(1000000))
    assert mix_stereo(pcm) == stereo
    assert len(FakeRenderer.instances) == 1
    await source.aclose()


async def test_sessions_are_independent() -> None:
    """Two playbacks of the same file own separate synths and frame cursors."""
    first, second = make_session(), make_session()
    first_source, second_source = first.stereo_stream(), second.stereo_stream()
    await anext(first_source)
    assert second.end_frame == 0
    await anext(second_source)
    await anext(first_source)
    assert first.end_frame == 2 * second.end_frame
    assert len(FakeRenderer.instances) == 2
    await asyncio.gather(first_source.aclose(), second_source.aclose())
    assert all(instance.closed for instance in FakeRenderer.instances)


async def test_render_error_wakes_sidecar_and_closes_synth() -> None:
    """Both consumers observe failure and native cleanup still completes."""
    session = make_session()
    consumer = session.attach()
    with patch.object(FakeRenderer, "render", side_effect=RuntimeError("synth failure")):
        source = session.stereo_stream()
        with pytest.raises(RuntimeError, match="synth failure"):
            await anext(source)
    assert session.state == "failed"
    assert FakeRenderer.instances[0].closed
    with pytest.raises(SessionClosedError, match="failed"):
        await consumer.read()


async def test_native_work_completes_before_cancellation_cleanup() -> None:
    """Cancelling an in-flight native call never deletes a synth still in use."""
    entered = threading.Event()
    release = threading.Event()
    original = FakeRenderer.render

    def slow_render(renderer: FakeRenderer, frames: int) -> NDArray[np.float32]:
        entered.set()
        assert release.wait(5)
        assert not renderer.closed
        return original(renderer, frames)

    session = make_session()
    source = session.stereo_stream()
    with patch.object(FakeRenderer, "render", slow_render):
        task = asyncio.create_task(anext(source))
        await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert session.state == "closed"
    assert FakeRenderer.instances[0].closed


async def test_real_file_timeline_is_not_retained_as_audio(midi_path: Path) -> None:
    """The parser's event timeline can be rendered without retaining the whole PCM piece."""
    midi = load_midi(str(midi_path))
    session = MidiSession("parsed", midi, "unused.sf2")
    async for _chunk in session.stereo_stream():
        assert session.retained_frames == 0
    assert session.end_frame == midi.total_frames
