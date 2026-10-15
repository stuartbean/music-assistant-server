"""Family selectors reference existing AirPlay endpoints without creating speakers."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.errors import InvalidDataError

from music_assistant.models.setup_flow import SetupSession
from music_assistant.providers.airplay.player import AirPlayPlayer
from music_assistant.providers.midi_player.config import family_entries
from music_assistant.providers.midi_player.provider import MidiPlayerProvider
from music_assistant.providers.midi_player.setup_flow import run_setup


def test_seven_optional_single_receiver_selectors() -> None:
    """Non-AirPlay devices are excluded and family assignments are single-select."""
    mass = MagicMock()
    client = MagicMock(spec=AirPlayPlayer)
    client.player_id = "speaker"
    client.display_name = "Office"
    client.protocol_parent_id = None
    mass.players.all_players.return_value = [client, MagicMock()]
    entries = family_entries(mass)
    assert len(entries) == 7
    assert {entry.key for entry in entries} == {
        "woodwind",
        "brass",
        "strings",
        "harp",
        "percussion",
        "organ",
        "voices",
    }
    assert all(not entry.required and not entry.multi_value for entry in entries)
    assert all([option.value for option in entry.options] == ["", "speaker"] for entry in entries)


async def test_setup_accepts_several_families_on_one_receiver() -> None:
    """The setup form preserves repeated receiver assignments across family keys."""
    session = MagicMock()
    session.context.setup_data = {}
    session.mass.players.all_players.return_value = []
    session.form = AsyncMock(return_value={"strings": "speaker", "harp": "speaker"})
    session.finish = AsyncMock()
    await run_setup(cast("SetupSession", session))
    session.finish.assert_awaited_once_with({"strings": "speaker", "harp": "speaker"})


async def test_empty_assignments_fail_before_registering_player() -> None:
    """An ensemble needs at least one selected receiver."""
    provider = MagicMock()
    provider.get_setup_value.return_value = ""
    with pytest.raises(InvalidDataError, match="at least one"):
        await MidiPlayerProvider.handle_async_init(provider)
