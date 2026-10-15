"""Register one virtual MIDI ensemble per provider instance."""

from __future__ import annotations

from music_assistant_models.errors import InvalidDataError

from music_assistant.models.player_provider import PlayerProvider
from music_assistant.providers.midi.routing import FAMILIES

from .player import MidiPlayer


class MidiPlayerProvider(PlayerProvider):
    """Configure MIDI family assignments without duplicating physical AirPlay players."""

    async def handle_async_init(self) -> None:
        """Require at least one configured family output."""
        if not any(self.get_setup_value(family.value) for family in FAMILIES):
            raise InvalidDataError("Assign at least one instrument family to an AirPlay player")

    async def discover_players(self) -> None:
        """Register the ensemble with a stable identity for this provider instance."""
        player_id = f"midi_player_{self.instance_id}"
        if self.mass.players.get_player(player_id) is None:
            await self.mass.players.register_or_update(MidiPlayer(self, player_id))
