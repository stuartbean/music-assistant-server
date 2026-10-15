"""Sample-aligned instrument-family mixing for individual AirPlay receivers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from music_assistant.providers.midi.constants import MIX_GAIN, PCM_CHANNELS, SAMPLE_RATE
from music_assistant.providers.midi.routing import FAMILIES

if TYPE_CHECKING:
    from music_assistant.providers.midi.routing import InstrumentFamily
    from music_assistant.providers.midi.session import PCMBlock


class FamilyMixer:
    """Mix selected stereo family pairs without changing frame counts or positions."""

    def __init__(
        self, assignments: dict[InstrumentFamily, str], *, fade_seconds: float = 0
    ) -> None:
        """Snapshot family assignments and an optional common playback-start fade."""
        self.routes: dict[str, list[int]] = {}
        for index, family in enumerate(FAMILIES):
            if player_id := assignments.get(family):
                self.routes.setdefault(player_id, []).append(index)
        self._fade_frames = round(fade_seconds * SAMPLE_RATE)
        self._frames = 0
        self._next_frame: int | None = None

    def mix(self, block: PCMBlock) -> dict[str, bytes]:
        """Return one stereo float32 block per receiver for this exact source interval."""
        if len(block.data) != block.frame_count * PCM_CHANNELS * 4:
            raise ValueError("Invalid family PCM block size")
        if self._next_frame is not None and block.start_frame != self._next_frame:
            raise ValueError("Noncontiguous MIDI family frames")
        self._next_frame = block.start_frame + block.frame_count
        pcm = np.frombuffer(block.data, dtype="<f4").reshape(block.frame_count, len(FAMILIES), 2)
        gain = None
        if self._frames < self._fade_frames:
            phase = np.minimum(
                (self._frames + np.arange(block.frame_count, dtype=np.float64)) / self._fade_frames,
                1.0,
            )
            gain = (0.5 - 0.5 * np.cos(np.pi * phase))[:, None]
        result = {}
        for player_id, families in self.routes.items():
            mixed = pcm[:, families, :].sum(axis=1, dtype=np.float64) * MIX_GAIN
            if gain is not None:
                mixed *= gain
            result[player_id] = np.clip(mixed, -1, 1).astype("<f4").tobytes()
        self._frames += block.frame_count
        return result
