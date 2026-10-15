"""Family mixing preserves positions, stereo separation and fixed headroom."""

import numpy as np
import pytest

from music_assistant.providers.midi.routing import FAMILIES, InstrumentFamily
from music_assistant.providers.midi.session import PCMBlock
from music_assistant.providers.midi.synthesizer import mix_stereo
from music_assistant.providers.midi_player.mixer import FamilyMixer


def block(start: int = 0, frames: int = 4096) -> PCMBlock:
    """Create distinct constant left/right signals on every family pair."""
    data = np.tile(np.arange(14, dtype="<f4") / 100, (frames, 1))
    return PCMBlock(start, frames, data.tobytes())


def test_all_families_on_one_speaker_match_reference() -> None:
    """Grouping every family reproduces MidiProvider's standard stereo mix exactly."""
    pcm = block(30 * 44100)
    mixer = FamilyMixer(dict.fromkeys(FAMILIES, "speaker"))
    assert mixer.mix(pcm)["speaker"] == mix_stereo(
        np.frombuffer(pcm.data, dtype="<f4").reshape(-1, 14)
    )


def test_families_sum_only_into_their_assigned_receiver() -> None:
    """One receiver gets multiple families while another receives a distinct stereo pair."""
    mixer = FamilyMixer(
        {InstrumentFamily.STRINGS: "a", InstrumentFamily.HARP: "a", InstrumentFamily.BRASS: "b"}
    )
    output = mixer.mix(block())
    a = np.frombuffer(output["a"], dtype="<f4").reshape(-1, 2)
    b = np.frombuffer(output["b"], dtype="<f4").reshape(-1, 2)
    np.testing.assert_allclose(
        a, np.tile([(0.04 + 0.06) / 16, (0.05 + 0.07) / 16], (4096, 1)), atol=1e-9
    )
    np.testing.assert_allclose(b, np.tile([0.02 / 16, 0.03 / 16], (4096, 1)), atol=1e-9)
    assert len(output["a"]) == len(output["b"]) == 4096 * 8


def test_seek_ramp_is_shared_and_spans_blocks() -> None:
    """The gain ramp follows playback frames, independently of absolute seek position."""
    mixer = FamilyMixer(
        {InstrumentFamily.STRINGS: "a", InstrumentFamily.BRASS: "b"}, fade_seconds=0.02
    )
    parts = []
    for start, count in ((10000, 400), (10400, 600)):
        data = np.ones((count, 14), dtype="<f4")
        output = mixer.mix(PCMBlock(start, count, data.tobytes()))
        assert output["a"] == output["b"]
        parts.append(np.frombuffer(output["a"], dtype="<f4").reshape(-1, 2))
    pcm = np.concatenate(parts)
    assert pcm[0, 0] == 0
    assert np.all(np.diff(pcm[:, 0]) >= 0)
    assert np.all(pcm[882:] == 1 / 16)


def test_discontinuous_or_malformed_blocks_fail_explicitly() -> None:
    """Dropped family frames cannot silently shift a speaker's timeline."""
    mixer = FamilyMixer({InstrumentFamily.STRINGS: "a"})
    mixer.mix(block())
    with pytest.raises(ValueError, match="Noncontiguous"):
        mixer.mix(block(5000))
    with pytest.raises(ValueError, match="size"):
        mixer.mix(PCMBlock(4096, 1, b""))
