"""Virtual-player ownership, exact queue association and warm seek lifecycle."""

from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.config_entries import ProviderConfig
from music_assistant_models.enums import PlaybackState, ProviderType
from music_assistant_models.errors import PlayerCommandFailed
from music_assistant_models.provider import ProviderManifest
from music_assistant_models.queue_item import QueueItem

from music_assistant.models.player import PlayerMedia
from music_assistant.providers.airplay.player import AirPlayPlayer
from music_assistant.providers.midi_player.player import MidiPlayer
from music_assistant.providers.midi_player.provider import MidiPlayerProvider
from tests.providers.midi.test_provider import make_provider
from tests.providers.midi.test_session import FakeRenderer


class FakeTransport:
    """Exercise source lifetimes and PCM callbacks without connecting physical speakers."""

    def __init__(
        self, _provider: Any, clients: list[Any], _format: Any, _media: Any, *, member_audio: Any
    ) -> None:
        """Capture the shared member list and routing callback."""
        self.media = _media
        self.sync_clients = clients
        self.member_audio = member_audio
        self.source: AsyncGenerator[bytes] | None = None
        self.start_time = 1.0
        self.effective_start_time = 1.0
        self.warm_success = True
        self.outputs: dict[str, bytes] = {}
        self.start_count = 0
        self.replace_count = 0
        for client in clients:
            client.stream = SimpleNamespace(session=self, running=True)

    def can_replace(self, _clients: Any, _format: Any) -> bool:
        """Allow the fake transport to exercise warm replacement."""
        return True

    async def start(self, source: AsyncGenerator[bytes]) -> None:
        """Consume and route the first frame interval."""
        self.source = source
        reference = await anext(source)
        self.outputs = {
            client.player_id: self.member_audio(client.player_id, reference)
            for client in self.sync_clients
        }
        self.start_count += 1

    async def replace(
        self, source: AsyncGenerator[bytes], _media: Any, *, member_audio: Any
    ) -> bool:
        """Close the previous generator and exercise a warm restart."""
        if self.source:
            await self.source.aclose()
        self.member_audio = member_audio
        self.media = _media
        await self.start(source)
        self.replace_count += 1
        return self.warm_success

    async def standby(self) -> bool:
        """Close rendering while preserving fake receiver connections."""
        if self.source:
            await self.source.aclose()
        return True

    async def stop(self) -> None:
        """Close rendering and release the fake receiver streams."""
        if self.source:
            await self.source.aclose()
        for client in self.sync_clients:
            if client.stream and client.stream.session is self:
                client.stream = None


def player_fixture(midi_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any, Any]:
    """Build a real virtual player with fake hardware and a real MIDI provider/session."""
    monkeypatch.setattr("music_assistant.providers.midi.session.MidiRenderer", FakeRenderer)
    monkeypatch.setattr(
        "music_assistant.providers.midi_player.player.AirPlayStreamSession", FakeTransport
    )
    midi = make_provider(midi_path)
    mass = cast("MagicMock", midi.mass)
    manifest = ProviderManifest(
        type=ProviderType.PLAYER,
        domain="midi_player",
        name="MidiPlayer",
        description="Test",
        codeowners=["@music-assistant"],
    )
    config = ProviderConfig(
        type=ProviderType.PLAYER,
        domain="midi_player",
        instance_id="ensemble",
        values={},
        setup_data={"strings": "speaker", "brass": "speaker"},
    )
    provider = MidiPlayerProvider(mass, manifest, config)
    mass.config.get.return_value = config.setup_data
    player = MidiPlayer(provider, "ensemble-player")
    monkeypatch.setattr(player, "update_state", MagicMock())
    client = MagicMock(spec=AirPlayPlayer)
    client.player_id = "speaker"
    client.provider_id = "airplay"
    client.provider = MagicMock()
    client.available = True
    client.protocol_parent_id = None
    client.state.active_group = None
    client.synced_to = None
    client.group_members = []
    client.stream = None
    mass.players.get_player.side_effect = lambda player_id, *_args, **_kwargs: {
        "speaker": client,
        "ensemble-player": player,
    }.get(player_id)
    mass.get_provider.return_value = midi
    state = SimpleNamespace(session_id="generation", seek_transition=None)
    mass.player_queues.queue_data_or_none.return_value = state
    return player, midi, client, state


async def start_item(
    player: MidiPlayer, midi: Any, state: Any, seek: int = 0
) -> tuple[QueueItem, PlayerMedia]:
    """Bind one queued MIDI item to the virtual player's current generation."""
    details = await midi.get_stream_details("performance")
    details.seek_position = seek
    item = QueueItem(
        queue_id=player.player_id,
        queue_item_id="item",
        name="Test",
        duration=4,
        streamdetails=details,
    )
    cast("MagicMock", player.mass.player_queues).get_item.return_value = item
    media = PlayerMedia(
        uri="test",
        source_id=player.player_id,
        queue_item_id="item",
        queue_session_id=state.session_id,
    )
    await player.play_media(media)
    return item, media


async def test_one_receiver_can_take_several_families(
    midi_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Several assignments deduplicate into one receiver and one transport stream."""
    player, midi, client, state = player_fixture(midi_path, monkeypatch)
    try:
        await start_item(player, midi, state)
        assert player.playback_state == PlaybackState.PLAYING
        assert player.group_members == ["speaker"]
        assert player.uses_audio_buffer is False
        assert player.flow_mode is False
        assert len(player._session.sync_clients) == 1
        assert player._playback.mixer.routes == {"speaker": [1, 2]}
        assert client.stream.session is player._session
    finally:
        await player.stop()
    assert not midi._sessions
    assert player.group_members == ["speaker"]
    assert player.live_session_members == []
    assert not player.is_active_session


async def test_seek_reuses_transport_but_gets_fresh_render(
    midi_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old stereo cache never supplies the seek; frame origin and generation are explicit."""
    player, midi, _client, state = player_fixture(midi_path, monkeypatch)
    try:
        item, media = await start_item(player, midi, state)
        transport = player._session
        original = player._playback.session
        assert item.streamdetails is not None
        item.streamdetails.seek_position = 1
        state.session_id = "seek"
        state.seek_transition = ("seek", "item")
        media.queue_session_id = "seek"
        await player.play_media(media)
        assert player._session is transport
        assert player._playback.session.session_id != original.session_id
        assert player._playback.session.first_frame == 44100
        assert transport.media.elapsed_time == 1
        assert original.state == "closed"
    finally:
        await player.stop()
    assert not midi._sessions


async def test_failed_warm_seek_restarts_with_an_unconsumed_source(
    midi_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partially consumed failed replacement cannot poison the cold fallback."""
    player, midi, _client, state = player_fixture(midi_path, monkeypatch)
    try:
        item, media = await start_item(player, midi, state)
        transport = player._session
        transport.warm_success = False
        assert item.streamdetails is not None
        item.streamdetails.seek_position = 1
        await player.play_media(media)
        assert player._session is not transport
        assert player._playback.session.first_frame == 44100
        assert player.playback_state == PlaybackState.PLAYING
        assert len(midi._sessions) == 1
    finally:
        await player.stop()
    assert not midi._sessions


async def test_pause_and_resume_close_and_replace_renderer(
    midi_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pause parks transport; a resume uses the normal queue resume path."""
    player, midi, _client, state = player_fixture(midi_path, monkeypatch)
    player.mass.player_queues.resume = AsyncMock()
    try:
        await start_item(player, midi, state)
        await player.pause()
        assert player.playback_state == PlaybackState.PAUSED
        assert not midi._sessions
        assert player.is_active_session
        await player.play()
        player.mass.player_queues.resume.assert_awaited_once_with(player.player_id)
    finally:
        await player.stop()


async def test_foreign_queue_generation_and_grouped_receivers_fail(
    midi_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale request cannot take over grouped receivers or bind another queue's render."""
    player, midi, client, state = player_fixture(midi_path, monkeypatch)
    client.state.active_group = "another-group"
    with pytest.raises(PlayerCommandFailed, match="existing group"):
        await start_item(player, midi, state)
    client.state.active_group = None
    details = await midi.get_stream_details("performance")
    player.mass.player_queues.get_item.return_value = QueueItem(
        queue_id=player.player_id,
        queue_item_id="item",
        name="Test",
        duration=4,
        streamdetails=details,
    )
    with pytest.raises(PlayerCommandFailed, match="generation"):
        await player.play_media(
            PlayerMedia(
                uri="test",
                source_id=player.player_id,
                queue_item_id="item",
                queue_session_id="stale",
            )
        )
    assert not midi._sessions


async def test_failed_cold_start_releases_every_resource(
    midi_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after attachment never leaves the ensemble holding its receiver or synth."""
    player, midi, client, state = player_fixture(midi_path, monkeypatch)
    monkeypatch.setattr(
        FakeTransport, "start", AsyncMock(side_effect=PlayerCommandFailed("connection failed"))
    )
    with pytest.raises(PlayerCommandFailed, match="connection failed"):
        await start_item(player, midi, state)
    assert not midi._sessions
    assert player._session is None
    assert player.group_members == ["speaker"]
    assert player.live_session_members == []
    assert client.stream is None
    assert player.playback_state == PlaybackState.IDLE


async def test_a_foreign_stream_is_not_warm_flushed_or_stopped(
    midi_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receiver takeover invalidates warm reuse and is not undone during old-session cleanup."""
    player, midi, client, state = player_fixture(midi_path, monkeypatch)
    _item, media = await start_item(player, midi, state)
    transport = player._session
    foreign = SimpleNamespace(session=object(), running=True)
    client.stream = foreign
    await player.stop()
    assert client.stream is foreign
    assert not midi._sessions
    await player.play_media(media)
    assert transport.replace_count == 0
    assert player._session is not transport
    await player.stop()
