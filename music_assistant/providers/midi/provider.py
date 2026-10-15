"""Music Assistant integration and exact-playback multichannel session discovery."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from music_assistant_models.enums import ContentType, MediaType, ProviderFeature, StreamType
from music_assistant_models.errors import AudioError, InvalidDataError, MediaNotFoundError
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

from .constants import CONF_MIDI_FILE, CONF_SOUNDFONT, DEFAULT_SOUNDFONT, MAX_SESSIONS, SAMPLE_RATE
from .midi_file import MidiFile, load_midi
from .session import MidiSession, SessionClosedError
from .synthesizer import FluidSynth

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant

TRACK_ID = "performance"
ARTIST_ID = "midi"


@dataclass
class MidiPlaybackRequest:
    """Server-only association stored on the exact StreamDetails MA gives the producer."""

    request_id: str
    session: MidiSession | None = None


class MidiProvider(MusicProvider):
    """Expose a configured MIDI file as a track and synthesize it on demand."""

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
        self._midi: MidiFile
        self._soundfont = ""

    @property
    def is_streaming_provider(self) -> bool:
        """Return False: each instance represents its own configured local file."""
        return False

    async def handle_async_init(self) -> None:
        """Validate the local MIDI file, SoundFont and native synth installation."""
        path = str(self.get_setup_value(CONF_MIDI_FILE) or "")
        self._soundfont = str(self.get_setup_value(CONF_SOUNDFONT) or DEFAULT_SOUNDFONT)
        self._midi = await asyncio.to_thread(self._load, path, self._soundfont)

    async def unload(self, is_removed: bool = False) -> None:
        """Close every active/pending render and attached consumer."""
        self.unloading = True
        await asyncio.gather(*(session.close() for session in tuple(self._sessions.values())))
        self._sessions.clear()

    async def get_library_tracks(self) -> AsyncGenerator[Track]:
        """Expose the configured performance in the MA library."""
        yield await self.get_track(TRACK_ID)

    async def get_track(self, prov_track_id: str) -> Track:
        """Return the configured MIDI performance."""
        if prov_track_id != TRACK_ID:
            raise MediaNotFoundError("Unknown MIDI performance")
        return Track(
            item_id=TRACK_ID,
            provider=self.instance_id,
            name=self._midi.name,
            duration=ceil(self._midi.total_frames / SAMPLE_RATE),
            artists=UniqueList([await self.get_artist(ARTIST_ID)]),
            provider_mappings={self._mapping(TRACK_ID)},
        )

    async def get_artist(self, prov_artist_id: str) -> Artist:
        """Return the generic artist of this MIDI performance."""
        if prov_artist_id != ARTIST_ID:
            raise MediaNotFoundError("Unknown MIDI artist")
        return Artist(
            item_id=ARTIST_ID,
            provider=self.instance_id,
            name="MIDI performance",
            provider_mappings={self._mapping(ARTIST_ID)},
        )

    async def browse(self, path: str) -> Sequence[MediaItemType | ItemMapping | BrowseFolder]:
        """Browse the configured performance."""
        return [await self.get_track(TRACK_ID)]

    async def search(
        self, search_query: str, media_types: list[MediaType], limit: int = 5
    ) -> SearchResults:
        """Search the configured performance by title."""
        results = SearchResults()
        if (
            MediaType.TRACK in media_types
            and limit > 0
            and search_query.casefold() in self._midi.name.casefold()
        ):
            results.tracks = [await self.get_track(TRACK_ID)]
        return results

    async def get_stream_details(
        self, item_id: str, media_type: MediaType = MediaType.TRACK
    ) -> StreamDetails:
        """Describe raw stereo PCM and reserve a unique association without creating a synth."""
        if media_type != MediaType.TRACK or item_id != TRACK_ID:
            raise MediaNotFoundError("Unknown MIDI performance")
        return StreamDetails(
            provider=self.instance_id,
            item_id=TRACK_ID,
            media_type=MediaType.TRACK,
            audio_format=self._audio_format(),
            stream_type=StreamType.CUSTOM,
            duration=ceil(self._midi.total_frames / SAMPLE_RATE),
            allow_seek=True,
            can_seek=True,
            data=MidiPlaybackRequest(uuid4().hex),
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
        if not isinstance(request, MidiPlaybackRequest):
            raise SessionClosedError("Stream details do not belong to a MIDI playback")
        if request.session is not None:
            return request.session
        if len(self._sessions) >= MAX_SESSIONS:
            raise AudioError("Too many MIDI render sessions; close unused pending attachments")
        first_frame = round(streamdetails.seek_position * SAMPLE_RATE)
        session = MidiSession(
            request.request_id,
            self._midi,
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
        if seek_position < 0 or seek_position > ceil(self._midi.total_frames / SAMPLE_RATE):
            raise AudioError("MIDI seek position is outside the performance")
        request = streamdetails.data
        if not isinstance(request, MidiPlaybackRequest):
            raise AudioError("Missing MIDI playback association")
        if request.session and request.session.state in ("eof", "closed", "failed"):
            # MA can reopen details after pause/seek. Every render has a new ID,
            # while existing consumers keep their original terminal session.
            streamdetails.data = MidiPlaybackRequest(uuid4().hex)
        session = self.get_playback_session(streamdetails)
        session.set_source_origin(min(seek_position * SAMPLE_RATE, self._midi.total_frames))
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

    def _load(self, path: str, soundfont: str) -> MidiFile:
        midi_path = Path(path)
        font_path = Path(soundfont)
        if not midi_path.is_file() or midi_path.stat().st_size > 16 * 1024 * 1024:
            raise InvalidDataError("Configure a readable MIDI file no larger than 16 MiB")
        if not font_path.is_file() or font_path.suffix.casefold() != ".sf2":
            raise InvalidDataError("Configure a readable General MIDI .sf2 SoundFont")
        midi = load_midi(path)
        synth = FluidSynth(soundfont)
        synth.close()
        return midi
