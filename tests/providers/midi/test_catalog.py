"""Bundled catalog discovery and exact per-track playback associations."""

from dataclasses import replace
from pathlib import Path

import pytest
from music_assistant_models.enums import MediaType
from music_assistant_models.errors import MediaNotFoundError

from music_assistant.providers.midi.catalog import MidiCatalogItem, load_catalog
from music_assistant.providers.midi.constants import CONF_MIDI_FILE
from music_assistant.providers.midi.session import SessionClosedError
from tests.providers.midi.test_provider import make_provider


async def test_bundled_sample_without_custom_file(midi_path: Path, soundfont: str) -> None:
    """Default setup exposes the shipped score and can produce real PCM immediately."""
    provider = make_provider(midi_path, soundfont)
    provider.config.setup_data.pop(CONF_MIDI_FILE)
    await provider.handle_async_init()
    try:
        tracks = list(await provider.browse("midi-test://"))
        assert len(tracks) == 1
        track = await provider.get_track("sample_holst_planets")
        assert tracks == [track]
        assert track.name == "The Planets, Op. 32"
        assert track.artists[0].name == "Gustav Holst"
        assert track.duration == 2743
        assert (await provider.search("holst", [MediaType.TRACK])).tracks == [track]
        assert [item async for item in provider.get_library_tracks()] == []
        details = await provider.get_stream_details(track.item_id)
        source = provider.get_audio_stream(details)
        try:
            assert len(await anext(source)) > 0
        finally:
            await source.aclose()
    finally:
        await provider.unload()


def test_custom_file_and_additional_samples(
    midi_path: Path, soundfont: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new packaged filename gets its own ID alongside the existing custom ID."""
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "another.mid").write_bytes(midi_path.read_bytes())
    monkeypatch.setattr("music_assistant.providers.midi.catalog.SAMPLES_DIR", samples)
    catalog = load_catalog(str(midi_path), soundfont)
    assert set(catalog) == {"performance", "sample_another"}
    assert catalog["performance"].midi == catalog["sample_another"].midi


async def test_each_playback_uses_its_selected_score(midi_path: Path) -> None:
    """Catalog items resolve their own timelines and cannot exchange playback associations."""
    provider = make_provider(midi_path)
    midi = provider._catalog["performance"].midi
    other = replace(midi, name="Other score", total_frames=midi.total_frames + 44100)
    provider._catalog["sample_other"] = MidiCatalogItem("sample_other", other)
    first = await provider.get_stream_details("performance")
    second = await provider.get_stream_details("sample_other")
    try:
        assert first.duration is not None
        assert second.duration == first.duration + 1
        first_session = provider.get_playback_session(first)
        second_session = provider.get_playback_session(second)
        assert first_session.total_frames == midi.total_frames
        assert second_session.total_frames == other.total_frames
        assert first_session.session_id != second_session.session_id
        second.data = first.data
        with pytest.raises(SessionClosedError, match="do not belong"):
            provider.get_playback_session(second)
        with pytest.raises(MediaNotFoundError):
            await provider.get_track("missing")
    finally:
        await provider.unload()
