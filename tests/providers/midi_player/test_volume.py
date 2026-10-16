"""MIDI ensemble volume uses MA's group controls and visible receiver identities."""

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from music_assistant_models.config_entries import PlayerConfig, ProviderConfig
from music_assistant_models.enums import PlaybackState, PlayerFeature, PlayerType, ProviderType
from music_assistant_models.provider import ProviderManifest

from music_assistant.controllers.players import PlayerController
from music_assistant.models.player import Player
from music_assistant.providers.airplay.stream_session import AirPlayStreamSession
from music_assistant.providers.midi_player.player import MidiPlayer
from music_assistant.providers.midi_player.provider import MidiPlayerProvider
from tests.common import MockPlayer, MockProvider
from tests.providers.midi.test_provider import make_provider


@pytest.fixture(params=[False, True], ids=["standalone", "linked"])
def volume_group(
    request: pytest.FixtureRequest, midi_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[PlayerController, MidiPlayer, list[MockPlayer], list[MockPlayer]]:
    """Build an idle MIDI group with two real member states and optional protocol children."""
    mass = cast("MagicMock", make_provider(midi_path).mass)
    mass.config.get_raw_player_config_value.side_effect = lambda _pid, _key, default=None: default
    mass.config.get_raw_core_config_value.return_value = "GLOBAL"
    mass.player_queues.get.return_value = None
    mass.closing = False
    mass.loop = None
    controller = PlayerController(mass)
    monkeypatch.setattr(controller, "get_config_value", MagicMock(return_value=5))
    mass.players = controller
    provider = MockProvider("receiver", mass=mass)
    members = [MockPlayer(provider, name, name) for name in ("a", "b")]
    outputs = members
    if request.param:
        outputs = [
            MockPlayer(provider, f"ap-{member.player_id}", member.name or "", PlayerType.PROTOCOL)
            for member in members
        ]
        for output, member in zip(outputs, members, strict=True):
            output.set_protocol_parent_id(member.player_id)
    setup_data = {
        "strings": outputs[0].player_id,
        "brass": outputs[0].player_id,
        "harp": outputs[1].player_id,
    }
    mass.config.get.return_value = setup_data
    midi_provider = MidiPlayerProvider(
        mass,
        ProviderManifest(
            type=ProviderType.PLAYER,
            domain="midi_player",
            name="MidiPlayer",
            description="Test",
            codeowners=["@music-assistant"],
        ),
        ProviderConfig(
            type=ProviderType.PLAYER,
            domain="midi_player",
            instance_id="ensemble",
            name="MidiPlayer",
            values={},
            setup_data=setup_data,
        ),
    )
    ensemble = MidiPlayer(midi_provider, "ensemble-player")
    players: list[Player] = [*members, *outputs, ensemble]
    controller._players = {player.player_id: player for player in players}
    for player in controller._players.values():
        player.set_config(
            PlayerConfig(
                provider=player.provider_id,
                player_id=player.player_id,
                player_type=player.type,
                enabled=True,
                values={},
            )
        )
        player.set_initialized()
        monkeypatch.setattr(
            player, "update_state", partial(player.update_state, signal_event=False)
        )
    for member, volume in zip(members, (60, 30), strict=True):
        member._attr_volume_level = volume
        member._attr_supported_features.add(PlayerFeature.VOLUME_MUTE)
        monkeypatch.setattr(
            member,
            "volume_set",
            AsyncMock(side_effect=lambda value, m=member: setattr(m, "_attr_volume_level", value)),
        )
        monkeypatch.setattr(member, "volume_mute", AsyncMock())
    for player in controller._players.values():
        player.update_state(force_update=True)
    ensemble.update_state(force_update=True)
    return controller, ensemble, members, outputs


async def test_standard_group_volume_preserves_balance_and_restores_from_zero(
    volume_group: tuple[PlayerController, MidiPlayer, list[MockPlayer], list[MockPlayer]],
) -> None:
    """The controller adjusts each visible receiver once, including before playback."""
    controller, ensemble, members, _outputs = volume_group
    assert ensemble.state.type == PlayerType.GROUP
    assert set(ensemble.state.group_members) == {"a", "b"}
    assert ensemble.state.group_volume == 60
    assert PlayerFeature.VOLUME_SET in ensemble.state.supported_features
    assert not ensemble.is_active_session
    assert ensemble.live_session_members == []

    await controller.cmd_volume_set(ensemble.player_id, 30)
    cast("AsyncMock", members[0].volume_set).assert_awaited_once_with(30)
    cast("AsyncMock", members[1].volume_set).assert_awaited_once_with(15)
    await controller.cmd_group_volume(ensemble.player_id, 0)
    for member in members:
        cast("AsyncMock", member.volume_set).assert_awaited_with(0)
    await controller.cmd_group_volume(ensemble.player_id, 60)
    cast("AsyncMock", members[0].volume_set).assert_awaited_with(60)
    cast("AsyncMock", members[1].volume_set).assert_awaited_with(30)
    await controller.cmd_volume_up(ensemble.player_id)
    cast("AsyncMock", members[0].volume_set).assert_awaited_with(65)
    cast("AsyncMock", members[1].volume_set).assert_awaited_with(39)
    await controller.cmd_volume_down(ensemble.player_id)
    cast("AsyncMock", members[0].volume_set).assert_awaited_with(60)
    cast("AsyncMock", members[1].volume_set).assert_awaited_with(30)


async def test_group_mute_reaches_visible_members(
    volume_group: tuple[PlayerController, MidiPlayer, list[MockPlayer], list[MockPlayer]],
) -> None:
    """Group mute uses the existing member controls, including linked native controls."""
    controller, ensemble, members, _outputs = volume_group
    assert PlayerFeature.VOLUME_MUTE in ensemble.state.supported_features
    await controller.cmd_volume_mute(ensemble.player_id, True)
    await controller.cmd_volume_mute(ensemble.player_id, False)
    for member in members:
        assert cast("AsyncMock", member.volume_mute).await_args_list == [call(True), call(False)]


@pytest.mark.parametrize("playback_state", [PlaybackState.PLAYING, PlaybackState.PAUSED])
async def test_live_members_control_volume_and_claim_only_the_active_receivers(
    volume_group: tuple[PlayerController, MidiPlayer, list[MockPlayer], list[MockPlayer]],
    playback_state: PlaybackState,
) -> None:
    """Dropped receivers leave the live group; dormant membership does not capture speakers."""
    controller, ensemble, members, outputs = volume_group
    assert all(member.state.active_group is None for member in members)
    ensemble._session = cast("AirPlayStreamSession", SimpleNamespace(sync_clients=[outputs[0]]))
    ensemble._attr_playback_state = playback_state
    ensemble.update_state(force_update=True)
    for member in members:
        member.update_state(force_update=True)
    assert ensemble.live_session_members == [outputs[0].player_id]
    assert ensemble.state.group_members == ["a"]
    assert members[0].state.active_group == ensemble.player_id
    assert members[1].state.active_group is None
    await controller.cmd_group_volume(ensemble.player_id, 30)
    cast("AsyncMock", members[0].volume_set).assert_awaited_once_with(30)
    cast("AsyncMock", members[1].volume_set).assert_not_awaited()
    ensemble._session = None
    ensemble._attr_playback_state = PlaybackState.IDLE
    ensemble.update_state(force_update=True)
    for member in members:
        member.update_state(force_update=True)
    assert set(ensemble.state.group_members) == {"a", "b"}
    assert all(member.state.active_group is None for member in members)


def test_members_without_volume_or_mute_do_not_advertise_controls(
    volume_group: tuple[PlayerController, MidiPlayer, list[MockPlayer], list[MockPlayer]],
) -> None:
    """Hidden protocol capabilities do not override the visible players' configured controls."""
    _controller, ensemble, members, _outputs = volume_group
    for member in members:
        member._attr_supported_features.clear()
        member.update_state(force_update=True)
    ensemble.update_state(force_update=True)
    assert PlayerFeature.VOLUME_SET not in ensemble.state.supported_features
    assert PlayerFeature.VOLUME_MUTE not in ensemble.state.supported_features
    assert ensemble.state.group_volume is None
