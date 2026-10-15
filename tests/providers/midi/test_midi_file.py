"""MIDI track merging and timing validation."""

from pathlib import Path

import mido
import pytest
from music_assistant_models.errors import InvalidDataError

from music_assistant.providers.midi.constants import SAMPLE_RATE
from music_assistant.providers.midi.midi_file import load_midi


@pytest.mark.parametrize("file_type", [0, 1])
def test_tempo_timeline(file_type: int, tmp_path: Path) -> None:
    """Tempo changes affect subsequent deltas with no per-track clock or rounding drift."""
    midi = mido.MidiFile(type=file_type, ticks_per_beat=480)
    tempo = [
        mido.MetaMessage("set_tempo", tempo=500000),
        mido.MetaMessage("set_tempo", tempo=1000000, time=480),
    ]
    notes = [
        mido.Message("note_on", channel=2, note=60, velocity=100),
        mido.Message("note_off", channel=2, note=60, time=960),
    ]
    if file_type == 0:
        midi.tracks.append(mido.MidiTrack([*tempo, notes[0].copy(time=0), notes[1].copy(time=480)]))
    else:
        midi.tracks.extend([mido.MidiTrack(tempo), mido.MidiTrack(notes)])
    path = tmp_path / "tempo.mid"
    midi.save(path)
    result = load_midi(str(path))
    assert result.events[-1].frame == round(1.5 * SAMPLE_RATE)
    assert result.total_frames == round(3.5 * SAMPLE_RATE)
    assert result.events[-1].channel == 2


def test_tracks_are_not_channels(midi_path: Path) -> None:
    """One MIDI track can contribute several channel pairs to the same timeline."""
    midi = load_midi(str(midi_path))
    assert {event.channel for event in midi.events if event.kind == "note_on"} == {0, 3, 9}
    assert midi.name == "Test performance"
    assert midi.end_frame == 2 * SAMPLE_RATE
    assert midi.total_frames == 4 * SAMPLE_RATE


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        (mido.Message("sysex", data=[0x43, 0x10, 0x4C, 0]), "SysEx"),
        (mido.MetaMessage("set_tempo", tempo=0), "tempo"),
        (mido.Message("control_change", control=126, value=1), "channel-mode"),
        (mido.MetaMessage("sequencer_specific", data=[1]), "metadata"),
    ],
)
def test_unsupported_events_are_rejected(
    message: mido.Message | mido.MetaMessage, reason: str, tmp_path: Path
) -> None:
    """Unsupported features fail before any audio is streamed."""
    midi = mido.MidiFile()
    midi.tracks.append(mido.MidiTrack([message, mido.Message("note_on", note=60)]))
    path = tmp_path / "unsupported.mid"
    midi.save(path)
    with pytest.raises(InvalidDataError, match=reason):
        load_midi(str(path))


@pytest.mark.parametrize(("file_type", "ticks", "reason"), [(2, 480, "Type"), (1, -24, "SMPTE")])
def test_unsupported_timing(file_type: int, ticks: int, reason: str, tmp_path: Path) -> None:
    """Asynchronous Type 2 and SMPTE division do not silently use PPQN timing."""
    midi = mido.MidiFile(type=file_type, ticks_per_beat=ticks)
    midi.tracks.append(mido.MidiTrack([mido.Message("note_on", note=60)]))
    path = tmp_path / "unsupported.mid"
    midi.save(path)
    with pytest.raises(InvalidDataError, match=reason):
        load_midi(str(path))


def test_gm_reset_and_running_status_zero_velocity(tmp_path: Path) -> None:
    """GM reset and note-on velocity zero survive parsing as explicit synth events."""
    midi = mido.MidiFile(type=0)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("sysex", data=[0x7E, 0x7F, 9, 1]),
                mido.Message("note_on", channel=9, note=36, velocity=80),
                mido.Message("note_on", channel=9, note=36, velocity=0, time=480),
            ]
        )
    )
    path = tmp_path / "reset.mid"
    midi.save(path)
    events = load_midi(str(path)).events
    assert events[0].kind == "reset"
    assert events[-1].param2 == 0
