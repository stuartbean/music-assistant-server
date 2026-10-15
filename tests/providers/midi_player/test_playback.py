"""The mixed reference drives one renderer and exact family intervals."""

from pathlib import Path

import numpy as np
import pytest

from music_assistant.providers.midi.routing import FAMILIES, InstrumentFamily
from music_assistant.providers.midi_player.playback import MidiPlayback
from tests.providers.midi.test_provider import make_provider
from tests.providers.midi.test_session import FakeRenderer


async def test_reference_and_family_mix_are_one_render(
    midi_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No synthesis occurs before consumption; all-family routing equals the reference."""
    monkeypatch.setattr("music_assistant.providers.midi.session.MidiRenderer", FakeRenderer)
    provider = make_provider(midi_path)
    details = await provider.get_stream_details("performance")
    details.seek_position = 1
    playback = MidiPlayback(provider, details, dict.fromkeys(FAMILIES, "a"))
    assert playback.session.state == "pending"
    source = playback.stream()
    try:
        reference = await anext(source)
        assert playback.member_audio("a", reference) == reference
        assert playback.session.first_frame == 44100
        assert playback.session.end_frame == 44100 + 4096
        assert provider.get_playback_session(details) is playback.session
        with pytest.raises(ValueError, match="outside"):
            playback.member_audio("a", b"foreign")
    finally:
        await source.aclose()
        await provider.unload()
    assert not provider._sessions


async def test_closing_an_unconsumed_playback_releases_pending_attachment(midi_path: Path) -> None:
    """Failed transport startup does not leak a reserved session or attached consumer."""
    provider = make_provider(midi_path)
    details = await provider.get_stream_details("performance")
    playback = MidiPlayback(provider, details, {InstrumentFamily.STRINGS: "a"})
    await playback.close()
    assert playback.session.state == "closed"
    assert not provider._sessions
    await provider.unload()


async def test_native_reference_routes_real_family_audio(midi_path: Path, soundfont: str) -> None:
    """The bridge drains real native rendering with two distinct receiver mixes."""
    provider = make_provider(midi_path, soundfont)
    await provider.handle_async_init()
    details = await provider.get_stream_details("performance")
    playback = MidiPlayback(
        provider,
        details,
        {InstrumentFamily.STRINGS: "strings", InstrumentFamily.PERCUSSION: "percussion"},
    )
    source = playback.stream()
    peaks = {"strings": 0.0, "percussion": 0.0}
    frames = 0
    try:
        async for reference in source:
            count = len(reference) // 8
            for player_id, peak in peaks.items():
                pcm = np.frombuffer(playback.member_audio(player_id, reference), dtype="<f4")
                assert len(pcm) == count * 2
                peaks[player_id] = max(peak, float(np.max(np.abs(pcm))))
            frames += count
        assert frames == playback.session.total_frames
        assert all(peak > 0.001 for peak in peaks.values())
    finally:
        await source.aclose()
        await provider.unload()
    assert not provider._sessions
