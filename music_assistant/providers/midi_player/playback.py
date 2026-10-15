"""Consume the standard MIDI stream and retrieve its exact family PCM intervals."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import TYPE_CHECKING

from .mixer import FamilyMixer

if TYPE_CHECKING:
    from music_assistant_models.streamdetails import StreamDetails

    from music_assistant.providers.midi.provider import MidiProvider
    from music_assistant.providers.midi.routing import InstrumentFamily


class MidiPlayback:
    """Bounded bridge from one standard stereo producer to per-receiver family mixes."""

    def __init__(
        self,
        provider: MidiProvider,
        details: StreamDetails,
        assignments: dict[InstrumentFamily, str],
        *,
        fade_seconds: float = 0,
    ) -> None:
        """Bind to fresh stream details before the standard producer is consumed."""
        self.provider = provider
        self.details = details
        self.session = provider.get_playback_session(details)
        self.consumer = self.session.attach()
        self.mixer = FamilyMixer(assignments, fade_seconds=fade_seconds)
        self._reference: bytes | None = None
        self._outputs: dict[str, bytes] = {}

    def member_audio(self, player_id: str, reference: bytes) -> bytes:
        """Retrieve the mix paired with the currently yielded standard stereo interval."""
        if reference is not self._reference:
            raise ValueError("AirPlay requested PCM outside the current MIDI interval")
        return self._outputs[player_id]

    async def stream(self) -> AsyncGenerator[bytes]:
        """Let standard stereo consumption drive the sole render, without playing the mix."""
        source = self.provider.get_audio_stream(self.details, int(self.details.seek_position))
        started = False
        try:
            async with aclosing(source):
                async for stereo in source:
                    if len(stereo) % 8:
                        raise ValueError("The MIDI reference ended inside a stereo frame")
                    frames = len(stereo) // 8
                    block = await self.consumer.read(frames)
                    if (
                        block is None
                        or block.frame_count != frames
                        or (not started and block.start_frame != self.session.first_frame)
                    ):
                        raise ValueError("MIDI reference and family PCM intervals differ")
                    self._outputs = self.mixer.mix(block)
                    self._reference = stereo
                    if not started:
                        await self.provider.on_stream_started(self.details)
                        started = True
                    self.details.seconds_streamed = (
                        self.session.end_frame - self.session.first_frame
                    ) / self.session.format.sample_rate
                    yield stereo
        finally:
            self.consumer.detach()
            self._outputs.clear()
            self._reference = None
            await self.session.close()
            if started:
                await self.provider.on_streamed(self.details)

    async def close(self) -> None:
        """Release a pending attachment if startup failed before consumption began."""
        self.consumer.detach()
        await self.session.close()
