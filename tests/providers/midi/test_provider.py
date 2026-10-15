"""Catalog, MA stream association and provider lifecycle tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from music_assistant_models.config_entries import ProviderConfig
from music_assistant_models.enums import MediaType, ProviderType
from music_assistant_models.errors import AudioError
from music_assistant_models.provider import ProviderManifest

from music_assistant.providers.midi.catalog import TRACK_ID, MidiCatalogItem
from music_assistant.providers.midi.constants import MAX_SESSIONS
from music_assistant.providers.midi.midi_file import load_midi
from music_assistant.providers.midi.provider import MidiProvider
from music_assistant.providers.midi.session import SessionClosedError
from tests.providers.midi.test_session import FakeRenderer


def make_provider(midi_path: Path, soundfont: str = "unused.sf2") -> MidiProvider:
    """Construct a real provider with a mock MA instance and a generated score."""
    manifest = ProviderManifest(
        type=ProviderType.MUSIC,
        domain="midi",
        name="MidiProvider",
        description="MIDI test provider",
        codeowners=["@music-assistant"],
    )
    config = ProviderConfig(
        type=ProviderType.MUSIC,
        domain="midi",
        instance_id="midi-test",
        values={},
        setup_data={"midi_file": str(midi_path), "soundfont": soundfont},
    )
    mass = MagicMock()
    mass.config.get.return_value = config.setup_data
    mass.config.decrypt_string.side_effect = lambda value: value
    provider = MidiProvider(mass, manifest, config)
    provider._catalog = {TRACK_ID: MidiCatalogItem(TRACK_ID, load_midi(str(midi_path)))}
    provider._soundfont = soundfont
    return provider


async def test_track_is_discoverable(midi_path: Path) -> None:
    """The configured file is available through library sync, browse, lookup and search."""
    provider = make_provider(midi_path)
    tracks = [track async for track in provider.get_library_tracks()]
    assert len(tracks) == 1
    assert tracks[0].name == "Test performance"
    assert tracks[0].duration == 4
    assert tracks[0].provider == "midi-test"
    assert list(await provider.browse("midi-test://")) == tracks
    assert (await provider.search("performance", [MediaType.TRACK])).tracks == tracks
    assert not (await provider.search("missing", [MediaType.TRACK])).tracks
    assert provider.is_streaming_provider is False


async def test_same_track_has_distinct_playback_requests(midi_path: Path) -> None:
    """Two players resolving the same track receive distinct sessions without any synthesis."""
    provider = make_provider(midi_path)
    first = await provider.get_stream_details("performance")
    second = await provider.get_stream_details("performance")
    first_session = provider.get_playback_session(first)
    second_session = provider.get_playback_session(second)
    assert first_session.session_id != second_session.session_id
    assert first_session.state == "pending"
    assert second_session.state == "pending"
    assert "data" not in first.to_dict()
    assert provider.get_playback_session(first) is first_session
    await provider.unload()
    assert str(first_session.state) == "closed"
    assert str(second_session.state) == "closed"
    with pytest.raises(SessionClosedError):
        provider.get_playback_session(first)


async def test_player_media_identifiers_resolve_exact_queue_generation(midi_path: Path) -> None:
    """Discovery checks queue/session identity rather than the shared provider item ID."""
    provider = make_provider(midi_path)
    details = await provider.get_stream_details("performance")
    details.queue_id = "player-a"
    details.queue_session_id = "generation-a"
    queues = cast("MagicMock", provider.mass.player_queues)
    queues.get_item.return_value = SimpleNamespace(streamdetails=details)
    session = provider.resolve_playback("player-a", "queue-item-a", "generation-a")
    assert session is provider.get_playback_session(details)
    queues.get_item.assert_called_with("player-a", "queue-item-a")
    with pytest.raises(SessionClosedError):
        provider.resolve_playback("player-a", "queue-item-a", "earlier-generation")
    with pytest.raises(SessionClosedError):
        provider.resolve_playback("player-b", "queue-item-a", "generation-a")
    await provider.unload()


async def test_reopened_generator_gets_new_session_id(
    midi_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pause/seek that closes the generator does not reuse its render identity."""
    monkeypatch.setattr("music_assistant.providers.midi.session.MidiRenderer", FakeRenderer)
    provider = make_provider(midi_path)
    details = await provider.get_stream_details("performance")
    original = provider.get_playback_session(details)
    consumer = original.attach()
    source = provider.get_audio_stream(details)
    await anext(source)
    await source.aclose()
    assert str(original.state) == "closed"
    replacement = provider.get_audio_stream(details)
    await anext(replacement)
    current = provider.get_playback_session(details)
    assert current.session_id != original.session_id
    with pytest.raises(SessionClosedError):
        await consumer.read()
    await provider.unload()
    await replacement.aclose()
    assert not provider._sessions


async def test_pending_session_registry_is_bounded(midi_path: Path) -> None:
    """Unused early attachments cannot create an unlimited registry of pending sessions."""
    provider = make_provider(midi_path)
    for _ in range(MAX_SESSIONS):
        details = await provider.get_stream_details("performance")
        provider.get_playback_session(details)
    details = await provider.get_stream_details("performance")
    with pytest.raises(AudioError, match="Too many MIDI"):
        provider.get_playback_session(details)
    await provider.unload()
    assert not provider._sessions


async def test_buffer_source_origin_can_precede_queue_seek(
    midi_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MA can refill stereo from zero while planning playback at a later seek point."""
    monkeypatch.setattr("music_assistant.providers.midi.session.MidiRenderer", FakeRenderer)
    provider = make_provider(midi_path)
    details = await provider.get_stream_details("performance")
    details.seek_position = 1
    session = provider.get_playback_session(details)
    consumer = session.attach()
    assert consumer.next_frame == 44100
    source = provider.get_audio_stream(details, seek_position=0)
    await anext(source)
    block = await consumer.read()
    assert block is not None
    assert block.start_frame == 0
    await source.aclose()
    await provider.unload()


@pytest.mark.parametrize("seek_position", [0, 1, 3])
async def test_seek_can_start_before_previous_generator_closes(
    midi_path: Path, monkeypatch: pytest.MonkeyPatch, seek_position: int
) -> None:
    """Overlapping buffer restarts own independent renders even when details are reused."""
    monkeypatch.setattr("music_assistant.providers.midi.session.MidiRenderer", FakeRenderer)
    provider = make_provider(midi_path)
    details = await provider.get_stream_details("performance")
    original = provider.get_playback_session(details)
    old_consumer = original.attach()
    source = provider.get_audio_stream(details)
    replacement = provider.get_audio_stream(details, seek_position=seek_position)
    try:
        await anext(source)
        assert original.state == "rendering"
        await anext(replacement)
        current = provider.get_playback_session(details)
        assert current.session_id != original.session_id
        assert current.first_frame == seek_position * 44100
        assert current.end_frame == seek_position * 44100 + 4096
        assert original.first_frame == 0
        assert original.end_frame == 4096
        assert old_consumer.session is original
        await source.aclose()
        assert str(original.state) == "closed"
        assert current.session_id in provider._sessions
        assert provider.get_playback_session(details) is current
        assert len(await anext(replacement)) == 4096 * 8
        assert current.end_frame == seek_position * 44100 + 8192
        with pytest.raises(SessionClosedError):
            await old_consumer.read()
    finally:
        await source.aclose()
        await replacement.aclose()
        await provider.unload()
    assert not provider._sessions
