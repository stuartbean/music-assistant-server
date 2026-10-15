"""Family assignment configuration using existing AirPlay endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType

from music_assistant.providers.airplay.player import AirPlayPlayer
from music_assistant.providers.midi.routing import FAMILIES

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant


def family_entries(mass: MusicAssistant) -> list[ConfigEntry]:
    """Return one optional AirPlay receiver selector for each instrument family."""
    options = [ConfigValueOption("", "Unassigned")]
    for player in mass.players.all_players(return_protocol_players=True):
        if not isinstance(player, AirPlayPlayer):
            continue
        parent = (
            mass.players.get_player(player.protocol_parent_id)
            if player.protocol_parent_id
            else None
        )
        name = parent.display_name if parent else player.display_name
        options.append(ConfigValueOption(player.player_id, name))
    return [
        ConfigEntry(
            key=family.value,
            type=ConfigEntryType.STRING,
            required=False,
            default_value="",
            options=options,
        )
        for family in FAMILIES
    ]
