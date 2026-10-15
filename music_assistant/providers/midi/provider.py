"""Music Assistant integration and exact-playback multichannel session discovery."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING
from uuid import uuid4

from music_assistant_models.enums import ContentType, MediaType, ProviderFeature, StreamType
from music_assistant_models.errors import AudioError, MediaNotFoundError
from music_assistant_models.media_items import (
    Artist,
    AudioFormat,
    BrowseFolder,
    ItemMapping,
    MediaItemType,
    ProviderMapping,
    SearchResults,
    Track,
    UniqueList,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.models.music_provider import MusicProvider

from .catalog import TRACK_ID, MidiCatalogItem, load_catalog
from .constants import CONF_MIDI_FILE, CONF_SOUNDFONT, DEFAULT_SOUNDFONT, MAX_SESSIONS, SAMPLE_RATE
from .session import MidiSession, SessionClosedError

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant


@dataclass
class MidiPlaybackRequest:
    """Server-only association stored on the exact StreamDetails MA gives the producer."""

    request_id: str
    item_id: str
    session: MidiSession | None = None


class MidiProvider(MusicProvider):
    """Expose bundled MIDI samples and an optional configured file as playable tracks."""

    def __init__(
        self, mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
    ) -> None:
        """Initialize the provider and its independent playback registry."""
        super().__init__(
            mass,
            manifest,
            config,
            {ProviderFeature.BROWSE, ProviderFeature.SEARCH, ProviderFeature.LIBRARY_TRACKS},
        )
        self._sessions: dict[str, MidiSession] = {}
        self._catalog: dict[str, MidiCatalogItem] = {}
        self._soundfont = ""

    @property
    def is_streaming_provider(self) -> bool:
        """Return False for the local MIDI catalog."""
        return False

    async def handle_async_init(self) -> None:
        """Load MIDI performances and validate the SoundFont and native synth installation."""
        path = str(self.get_setup_value(CONF_MIDI_FILE) or "")
        self._soundfont = str(self.get_setup_value(CONF_SOUNDFONT) or DEFAULT_SOUNDFONT)
        self._catalog = await asyncio.to_thread(load_catalog, path, self._soundfont)

    async def unload(self, is_removed: bool = False) -> None:
        """Close every active/pending render and attached consumer."""
        self.unloading = True
        await asyncio.gather(*(session.close() for session in tuple(self._sessions.values())))
        self._sessions.clear()

    async def get_library_tracks(self) -> AsyncGenerator[Track]:
        """Expose the configured performance in the MA library."""
        if TRACK_ID in self._catalog:
            yield await self.get_track(TRACK_ID)

    async def get_track(self, prov_track_id: str) -> Track:
        """Return a bundled sample or the configured MIDI performance."""
        item = self._get_catalog_item(prov_track_id)
        return Track(
            item_id=item.item_id,
            provider=self.instance_id,
            name=item.midi.name,
            duration=ceil(item.midi.total_frames / SAMPLE_RATE),
            artists=UniqueList([await self.get_artist(item.artist_id)]),
            provider_mappings={self._mapping(item.item_id)},
        )

    async def get_artist(self, prov_artist_id: str) -> Artist:
        """Return the composer or generic artist of a catalog performance."""
        for item in self._catalog.values():
            if item.artist_id == prov_artist_id:
                return Artist(
                    item_id=item.artist_id,
                    provider=self.instance_id,
                    name=item.artist_name,
                    provider_mappings={self._mapping(item.artist_id)},
                )
        raise MediaNotFoundError("Unknown MIDI artist")

    async def browse(self, path: str) -> Sequence[MediaItemType | ItemMapping | BrowseFolder]:
        """Browse every bundled sample and the optional custom performance."""
        return [await self.get_track(item_id) for item_id in self._catalog]

    async def search(
        self, search_query: str, media_types: list[MediaType], limit: int = 5
    ) -> SearchResults:
        """Search bundled and custom MIDI performances by title or composer."""
        results = SearchResults()
        if MediaType.TRACK in media_types and limit > 0:
            query = search_query.casefold()
            matches = [
                item.item_id
                for item in self._catalog.values()
                if query in item.midi.name.casefold() or query in item.artist_name.casefold()
            ]
            results.tracks = [await self.get_track(item_id) for item_id in matches[:limit]]
        return results

    async def get_stream_details(
        self, item_id: str, media_type: MediaType = MediaType.TRACK
    ) -> StreamDetails:
        """Describe raw stereo PCM and reserve a unique association without creating a synth."""
        if media_type != MediaType.TRACK:
            raise MediaNotFoundError("Unknown MIDI performance")
        item = self._get_catalog_item(item_id)
        return StreamDetails(
            provider=self.instance_id,
            item_id=item.item_id,
            media_type=MediaType.TRACK,
            audio_format=self._audio_format(),
            stream_type=StreamType.CUSTOM,
            duration=ceil(item.midi.total_frames / SAMPLE_RATE),
            allow_seek=True,
            can_seek=True,
            data=MidiPlaybackRequest(uuid4().hex, item_id),
        )

    def get_playback_session(self, streamdetails: StreamDetails) -> MidiSession:
        """
        Obtain the exact playback's session without starting or advancing synthesis.

        Attach before MA consumes stereo to retain audio from the seek origin.
        StreamDetails.data is server-only and is passed unchanged to get_audio_stream.
        Never resolve a playback by track ID alone.

        :param streamdetails: The exact queue item's resolved stream details.
        :raises SessionClosedError: When the provider is unloading or the request is foreign.
        """
        if self.unloading or streamdetails.provider != self.instance_id:
            raise SessionClosedError("The MIDI provider is unavailable for this playback")
        request = streamdetails.data
        if not isinstance(request, MidiPlaybackRequest) or request.item_id != streamdetails.item_id:
            raise SessionClosedError("Stream details do not belong to a MIDI playback")
        if request.session is not None:
            return request.session
        if len(self._sessions) >= MAX_SESSIONS:
            raise AudioError("Too many MIDI render sessions; close unused pending attachments")
        first_frame = round(streamdetails.seek_position * SAMPLE_RATE)
        session = MidiSession(
            request.request_id,
            self._get_catalog_item(request.item_id).midi,
            self._soundfont,
            first_frame,
            on_idle=lambda: self._discard_session(request.request_id),
        )
        request.session = session
        self._sessions[session.session_id] = session
        return session

    def resolve_playback(
        self, queue_id: str, queue_item_id: str, queue_session_id: str
    ) -> MidiSession:
        """
        Discover a session from the future player's PlayerMedia playback identifiers.

        :param queue_id: PlayerMedia.source_id (the queue owning playback).
        :param queue_item_id: PlayerMedia.queue_item_id, not the provider track ID.
        :param queue_session_id: PlayerMedia.queue_session_id, guarding seek/restart races.
        :raises SessionClosedError: When that exact queue generation no longer owns the audio.
        """
        item = self.mass.player_queues.get_item(queue_id, queue_item_id)
        details = item.streamdetails if item else None
        if (
            details is None
            or details.queue_id != queue_id
            or details.queue_session_id != queue_session_id
        ):
            raise SessionClosedError("The requested MIDI queue playback is no longer current")
        return self.get_playback_session(details)

    async def get_audio_stream(
        self, streamdetails: StreamDetails, seek_position: int = 0
    ) -> AsyncGenerator[bytes]:
        """Yield normal MA stereo while the same frames feed any attached multichannel reader."""
        midi = self._get_catalog_item(streamdetails.item_id).midi
        if seek_position < 0 or seek_position > ceil(midi.total_frames / SAMPLE_RATE):
            raise AudioError("MIDI seek position is outside the performance")
        request = streamdetails.data
        if not isinstance(request, MidiPlaybackRequest) or request.item_id != streamdetails.item_id:
            raise AudioError("Missing MIDI playback association")
        if request.session and request.session.state in ("eof", "closed", "failed"):
            # MA can reopen details after pause/seek. Every render has a new ID,
            # while existing consumers keep their original terminal session.
            streamdetails.data = MidiPlaybackRequest(uuid4().hex, streamdetails.item_id)
        session = self.get_playback_session(streamdetails)
        session.set_source_origin(min(seek_position * SAMPLE_RATE, midi.total_frames))
        async with aclosing(session.stereo_stream()) as source:
            async for chunk in source:
                yield chunk

    def _discard_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _mapping(self, item_id: str) -> ProviderMapping:
        return ProviderMapping(
            item_id=item_id,
            provider_domain=self.domain,
            provider_instance=self.instance_id,
            audio_format=self._audio_format(),
        )

    def _audio_format(self) -> AudioFormat:
        return AudioFormat(
            content_type=ContentType.PCM_F32LE, sample_rate=SAMPLE_RATE, bit_depth=32, channels=2
        )

    def _get_catalog_item(self, item_id: str) -> MidiCatalogItem:
        if item := self._catalog.get(item_id):
            return item
        raise MediaNotFoundError("Unknown MIDI performance")
