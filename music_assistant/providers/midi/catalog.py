"""Bundled samples and the optional custom MIDI performance."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from music_assistant_models.errors import InvalidDataError

from .midi_file import MidiFile, load_midi
from .synthesizer import FluidSynth

TRACK_ID = "performance"
ARTIST_ID = "midi"
SAMPLES_DIR = Path(__file__).with_name("samples")


@dataclass(frozen=True)
class MidiCatalogItem:
    """A playable MIDI score with stable catalog and artist identifiers."""

    item_id: str
    midi: MidiFile
    artist_id: str = ARTIST_ID
    artist_name: str = "MIDI performance"


def load_catalog(custom_path: str, soundfont: str) -> dict[str, MidiCatalogItem]:
    """
    Load the bundled catalog and optionally the configured file, validating synthesis support.

    :param custom_path: Optional custom file; an empty string selects bundled samples only.
    :param soundfont: General MIDI SoundFont for every performance in this provider instance.
    """
    font_path = Path(soundfont)
    if not font_path.is_file() or font_path.suffix.casefold() != ".sf2":
        raise InvalidDataError("Configure a readable General MIDI .sf2 SoundFont")
    catalog: dict[str, MidiCatalogItem] = {}
    if custom_path:
        catalog[TRACK_ID] = MidiCatalogItem(TRACK_ID, _load_file(Path(custom_path)))
    for path in sorted(SAMPLES_DIR.glob("*.mid")):
        item_id = f"sample_{path.stem}"
        midi = _load_file(path)
        if path.stem == "holst_planets":
            catalog[item_id] = MidiCatalogItem(
                item_id,
                replace(midi, name="The Planets, Op. 32"),
                "gustav_holst",
                "Gustav Holst",
            )
        else:
            catalog[item_id] = MidiCatalogItem(item_id, midi)
    if not catalog:
        raise InvalidDataError(
            "No bundled MIDI samples are installed; configure a custom MIDI file"
        )
    synth = FluidSynth(soundfont)
    synth.close()
    return catalog


def _load_file(path: Path) -> MidiFile:
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise InvalidDataError("Configure a readable MIDI file no larger than 16 MiB")
    return load_midi(str(path))
