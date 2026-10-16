"""One logical MIDI player coordinating different family mixes on AirPlay receivers."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from music_assistant_models.enums import ContentType, PlaybackState, PlayerFeature, PlayerType
from music_assistant_models.errors import PlayerCommandFailed
from music_assistant_models.media_items import AudioFormat

from music_assistant.models.player import Player, PlayerMedia
from music_assistant.providers.airplay.helpers import cli_clock_time
from music_assistant.providers.airplay.player import AirPlayPlayer
from music_assistant.providers.airplay.provider import AirPlayProvider
from music_assistant.providers.airplay.stream_session import AirPlayStreamSession
from music_assistant.providers.midi.constants import SAMPLE_RATE
from music_assistant.providers.midi.provider import MidiProvider
from music_assistant.providers.midi.routing import FAMILIES, InstrumentFamily

from .playback import MidiPlayback

if TYPE_CHECKING:
    from music_assistant_models.queue_item import QueueItem

    from .provider import MidiPlayerProvider


class MidiPlayer(Player):
    """A fixed-membership ensemble driven by a single MIDI renderer and AirPlay timeline."""

    _attr_type = PlayerType.GROUP
    _attr_name = "MIDI Ensemble"
    _attr_needs_poll = True
    _attr_poll_interval = 1

    def __init__(self, provider: MidiPlayerProvider, player_id: str) -> None:
        """Initialize the virtual player without claiming configured receivers."""
        self._session: AirPlayStreamSession | None = None
        super().__init__(provider, player_id)
        self._playback: MidiPlayback | None = None
        self._lock = asyncio.Lock()
        self._parents: dict[str, tuple[str, str | None]] = {}

    @property
    def supported_features(self) -> set[PlayerFeature]:
        """Expose playback controls and the volume controls supported by visible members."""
        features = {PlayerFeature.PLAY_MEDIA, PlayerFeature.PAUSE}
        for member in self._translate_protocol_ids_to_visible(set(self.group_members)):
            if member.state.available:
                features.update(
                    {PlayerFeature.VOLUME_SET, PlayerFeature.VOLUME_MUTE}
                    & member.state.supported_features
                )
        return features

    @property
    def group_members(self) -> list[str]:
        """Return live receivers during a session and configured receivers while idle."""
        if self._session is not None:
            return self.live_session_members
        return list(
            dict.fromkeys(
                str(value)
                for family in FAMILIES
                if (value := self.provider.get_setup_value(family.value))
            )
        )

    @property
    def live_session_members(self) -> list[str]:
        """Return receivers held by the playing or parked AirPlay session."""
        if self._session is None:
            return []
        return [player.player_id for player in self._session.sync_clients]

    @property
    def uses_audio_buffer(self) -> bool:
        """Consume fresh provider audio so family frames cannot be lost to stereo prebuffering."""
        return False

    @property
    def requires_flow_mode(self) -> bool:
        """Return False because the ensemble advances through ordinary queue item starts."""
        return False

    @property
    def is_active_session(self) -> bool:
        """Claim receivers only while a playing or parked session exists."""
        return self._session is not None

    async def play_media(self, media: PlayerMedia) -> None:
        """Start or seek a MIDI queue item with one family mix per receiver."""
        async with self._lock:
            clients, assignments = self._resolve_outputs()
            item, provider = self._resolve_item(media)
            playback = await self._prepare_playback(item, provider, media, assignments)
            source = playback.stream()
            transport_media = replace(media, elapsed_time=int(playback.details.seek_position))
            pcm_format = AudioFormat(
                content_type=ContentType.PCM_F32LE,
                sample_rate=SAMPLE_RATE,
                bit_depth=32,
                channels=2,
            )
            self._attr_current_media = media
            self._attr_active_source = self.player_id
            self._attr_elapsed_time = 0
            self._attr_elapsed_time_last_updated = time.time()
            try:
                if (
                    self._session
                    and all(
                        client.stream is not None and client.stream.session is self._session
                        for client in clients
                    )
                    and self._session.can_replace(clients, pcm_format)
                ):
                    if await self._session.replace(
                        source, transport_media, member_audio=playback.member_audio
                    ):
                        self._playback = playback
                        self._started(item)
                        return
                    await self._stop_session()
                    await source.aclose()
                    await playback.close()
                    playback = await self._prepare_playback(item, provider, media, assignments)
                    source = playback.stream()
                await self._stop_session()
                self._claim_outputs(clients)
                self._session = AirPlayStreamSession(
                    cast("AirPlayProvider", clients[0].provider),
                    clients,
                    pcm_format,
                    transport_media,
                    member_audio=playback.member_audio,
                )
                self._playback = playback
                self.update_state()
                await self._session.start(source)
                self._started(item)
            except BaseException:
                await self._stop_session()
                await source.aclose()
                await playback.close()
                self._attr_playback_state = PlaybackState.IDLE
                self.update_state()
                raise

    async def play(self) -> None:
        """Resume through a fresh coordinated render at the queue's parked position."""
        await self.mass.player_queues.resume(self.player_id)

    async def pause(self) -> None:
        """Park every receiver together, closing the current renderer."""
        async with self._lock:
            if self._session is None:
                return
            self._update_position()
            if not await self._session.standby():
                await self._stop_session()
            self._attr_playback_state = PlaybackState.PAUSED
            self.update_state()

    async def stop(self) -> None:
        """Stop the shared transport and release every receiver and MIDI attachment."""
        async with self._lock:
            await self._stop_session()
            self._attr_playback_state = PlaybackState.IDLE
            self.update_state()

    async def poll(self) -> None:
        """Report the anchored audible position and detect transport EOF/disconnection."""
        if self._session is None or self.playback_state != PlaybackState.PLAYING:
            return
        self._update_position()
        if not any(
            player.stream and player.stream.running and player.stream.session is self._session
            for player in self._session.sync_clients
        ):
            await self.stop()
        else:
            self.update_state()

    async def on_unload(self) -> None:
        """Release playback when this virtual player is unloaded."""
        try:
            await self.stop()
        finally:
            await super().on_unload()

    async def _prepare_playback(
        self,
        item: QueueItem,
        provider: MidiProvider,
        media: PlayerMedia,
        assignments: dict[InstrumentFamily, str],
    ) -> MidiPlayback:
        previous = item.streamdetails
        assert previous is not None
        # A stereo-only cache or old render cannot supply the new family's frames.
        # Reserve fresh details before attaching, keeping this queue generation exact.
        details = await provider.get_stream_details(previous.item_id)
        details.queue_id = media.source_id
        details.queue_session_id = media.queue_session_id
        details.seek_position = previous.seek_position
        details.fade_in = previous.fade_in
        item.streamdetails = details
        queue_data = self.mass.player_queues.queue_data_or_none(self.player_id)
        seeking = bool(
            queue_data
            and queue_data.seek_transition == (media.queue_session_id, item.queue_item_id)
        )
        return MidiPlayback(
            provider,
            details,
            assignments,
            fade_seconds=3 if previous.fade_in else 0.02 if seeking else 0,
        )

    def _resolve_outputs(self) -> tuple[list[AirPlayPlayer], dict[InstrumentFamily, str]]:
        assignments = {}
        clients: dict[str, AirPlayPlayer] = {}
        for family in FAMILIES:
            if not (value := self.provider.get_setup_value(family.value)):
                continue
            player_id = str(value)
            player = self.mass.players.get_player(player_id)
            if not isinstance(player, AirPlayPlayer) or not player.available:
                raise PlayerCommandFailed(f"The {family.value} AirPlay receiver is unavailable")
            parent = (
                self.mass.players.get_player(player.protocol_parent_id)
                if player.protocol_parent_id
                else player
            )
            owner = parent.state.active_group if parent else None
            own_session = player.stream is not None and player.stream.session is self._session
            if (owner and owner != self.player_id) or (
                not own_session and (player.synced_to or player.group_members)
            ):
                raise PlayerCommandFailed(
                    f"Remove {player.display_name} from its existing group first"
                )
            assignments[family] = player_id
            clients[player_id] = player
        if not clients:
            raise PlayerCommandFailed("Assign at least one instrument family to an AirPlay player")
        if len({player.provider_id for player in clients.values()}) != 1:
            raise PlayerCommandFailed("All receivers must use the same AirPlay provider instance")
        return list(clients.values()), assignments

    def _resolve_item(self, media: PlayerMedia) -> tuple[QueueItem, MidiProvider]:
        if media.source_id != self.player_id or not media.queue_item_id:
            raise PlayerCommandFailed("MidiPlayer requires a MIDI track from its own MA queue")
        queue_data = self.mass.player_queues.queue_data_or_none(self.player_id)
        if queue_data is None or queue_data.session_id != media.queue_session_id:
            raise PlayerCommandFailed("The MIDI queue generation has changed")
        item = self.mass.player_queues.get_item(self.player_id, media.queue_item_id)
        if item is None or item.streamdetails is None:
            raise PlayerCommandFailed("The MIDI queue item is no longer available")
        provider = self.mass.get_provider(item.streamdetails.provider)
        if not isinstance(provider, MidiProvider):
            raise PlayerCommandFailed("MidiPlayer can only play tracks supplied by MidiProvider")
        if item.extra_attributes.get("playback_speed", 1.0) != 1.0:
            raise PlayerCommandFailed("MidiPlayer currently supports normal playback speed only")
        return item, provider

    def _claim_outputs(self, clients: list[AirPlayPlayer]) -> None:
        for player in clients:
            if player.protocol_parent_id and (
                parent := self.mass.players.get_player(player.protocol_parent_id)
            ):
                self._parents[parent.player_id] = (player.player_id, parent.active_output_protocol)
                parent.set_active_output_protocol(player.player_id)

    def _started(self, item: QueueItem) -> None:
        self._attr_playback_state = PlaybackState.PLAYING
        self.mass.player_queues.track_loaded_in_buffer(self.player_id, item.queue_item_id)
        self.update_state()
        if self._session:
            for player in self._session.sync_clients:
                player.update_state()

    def _update_position(self) -> None:
        if self._session is not None and self._session.start_time > 0:
            self._attr_elapsed_time = max(0, cli_clock_time() - self._session.effective_start_time)
            self._attr_elapsed_time_last_updated = time.time()

    async def _stop_session(self) -> None:
        session, self._session = self._session, None
        playback, self._playback = self._playback, None
        try:
            if session:
                await session.stop()
        finally:
            if playback:
                await playback.close()
            for parent_id, (protocol_id, previous) in self._parents.items():
                endpoint = self.mass.players.get_player(protocol_id)
                if (
                    isinstance(endpoint, AirPlayPlayer)
                    and endpoint.stream is not None
                    and endpoint.stream.session is not session
                ):
                    continue
                if (parent := self.mass.players.get_player(parent_id)) and (
                    parent.active_output_protocol == protocol_id
                ):
                    parent.set_active_output_protocol(previous)
            self._parents.clear()
