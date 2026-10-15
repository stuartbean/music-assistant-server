"""General MIDI music provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .provider import MidiProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> MidiProvider:
    """Create the MIDI provider instance."""
    return MidiProvider(mass, manifest, config)
