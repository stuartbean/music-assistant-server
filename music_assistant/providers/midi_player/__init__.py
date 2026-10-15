"""MIDI instrument-family player using the existing AirPlay transport."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .provider import MidiPlayerProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> MidiPlayerProvider:
    """Create a configurable MIDI ensemble."""
    return MidiPlayerProvider(mass, manifest, config)
