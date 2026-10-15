"""Small MIDI files and optional native-synth assets."""

from __future__ import annotations

import ctypes
import os
from ctypes.util import find_library
from pathlib import Path

import mido
import pytest


@pytest.fixture
def midi_path(tmp_path: Path) -> Path:
    """Create a GM performance with program changes, sustain, pan and drums."""
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("track_name", name="Test performance"),
                mido.MetaMessage("set_tempo", tempo=500000),
                mido.MetaMessage("end_of_track", time=1920),
            ]
        )
    )
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("program_change", channel=0, program=40),
                mido.Message("control_change", channel=0, control=64, value=127),
                mido.Message("control_change", channel=0, control=10, value=0),
                mido.Message("note_on", channel=0, note=60, velocity=100),
                mido.Message("note_off", channel=0, note=60, time=480),
                mido.Message("control_change", channel=0, control=64, value=0, time=960),
                mido.Message("program_change", channel=0, program=0),
                mido.Message("note_on", channel=0, note=64, velocity=80),
                mido.Message("note_off", channel=0, note=64, time=480),
            ]
        )
    )
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("note_on", channel=9, note=36, velocity=100),
                mido.Message("note_off", channel=9, note=36, time=120),
                mido.Message("note_on", channel=3, note=67, velocity=80, time=120),
                mido.Message("pitchwheel", channel=3, pitch=2000),
                mido.Message("control_change", channel=3, control=11, value=80),
                mido.Message("note_off", channel=3, note=67, time=240),
            ]
        )
    )
    path = tmp_path / "performance.mid"
    midi.save(path)
    return path


@pytest.fixture
def soundfont() -> str:
    """Find an explicitly supplied or system-installed GM SoundFont for native tests."""
    try:
        ctypes.CDLL(find_library("fluidsynth") or "libfluidsynth.so.3")
    except OSError:
        pytest.skip("Native synthesis tests need libfluidsynth 2.3+")
    candidates = [
        os.environ.get("MIDI_TEST_SOUNDFONT", ""),
        "/usr/share/sounds/sf2/TimGM6mb.sf2",
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/soundfonts/default.sf2",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    pytest.skip("Native synthesis tests need libfluidsynth and MIDI_TEST_SOUNDFONT")
