"""MIDI rendering and configuration constants."""

from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType

from .routing import FAMILIES

SAMPLE_RATE = 44100
BLOCK_FRAMES = 4096
MIDI_CHANNELS = 16
PCM_CHANNELS = len(FAMILIES) * 2
RETAINED_FRAMES = SAMPLE_RATE * 2
MAX_SESSIONS = 16
MAX_CONSUMERS = 4
MAX_MIDI_FILE_BYTES = 16 * 1024 * 1024
MAX_MIDI_EVENTS = 500_000
MAX_MIDI_METADATA_BYTES = 1024 * 1024
SEEK_PREROLL_SECONDS = 2
RELEASE_SECONDS = 2
MIX_GAIN = 1 / 16
CONF_MIDI_FILE = "midi_file"
CONF_SOUNDFONT = "soundfont"
DEFAULT_SOUNDFONT = "/usr/share/sounds/sf2/TimGM6mb.sf2"
CONFIG_ENTRIES = (
    ConfigEntry(key=CONF_MIDI_FILE, type=ConfigEntryType.STRING, required=False),
    ConfigEntry(key=CONF_SOUNDFONT, type=ConfigEntryType.STRING, required=False, advanced=True),
)
