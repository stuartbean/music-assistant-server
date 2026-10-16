"""MIDI track merging and timing validation."""

from pathlib import Path
from struct import pack
from unittest.mock import MagicMock

import mido
import pytest
from music_assistant_models.errors import InvalidDataError

from music_assistant.providers.midi import midi_file
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


def test_event_limit_counts_running_status_and_metadata_before_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A small file cannot allocate an excessive decoded graph through running status."""
    monkeypatch.setattr(midi_file, "MAX_MIDI_EVENTS", 3)
    path = tmp_path / "running.mid"
    track = b"\x00\x90\x3c\x64\x00\x3c\x00\x00\xff\x2f\x00"
    _write_tracks(path, track)
    assert len(load_midi(str(path)).events) == 2
    _write_tracks(path, b"\x00\xff\x01\x00" + track)
    decoder = MagicMock(side_effect=AssertionError("Mido must not decode excessive events"))
    monkeypatch.setattr(mido, "MidiFile", decoder)
    with pytest.raises(InvalidDataError, match="3-event limit"):
        load_midi(str(path))
    decoder.assert_not_called()


def test_event_limit_is_shared_across_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Splitting a performance across tracks cannot multiply the event budget."""
    monkeypatch.setattr(midi_file, "MAX_MIDI_EVENTS", 3)
    track = b"\x00\x90\x3c\x64\x00\xff\x2f\x00"
    path = tmp_path / "tracks.mid"
    _write_tracks(path, track, track)
    with pytest.raises(InvalidDataError, match="event limit"):
        load_midi(str(path))


@pytest.mark.parametrize("prefix", [b"\xff\x01", b"\xf0"])
def test_payload_limit_precedes_metadata_or_sysex_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: bytes
) -> None:
    """Large text and SysEx payloads are bounded before Mido expands them into objects."""
    monkeypatch.setattr(midi_file, "MAX_MIDI_METADATA_BYTES", 4)
    path = tmp_path / "payload.mid"
    _write_tracks(path, b"\x00" + prefix + b"\x05hello")
    decoder = MagicMock(side_effect=AssertionError("Mido must not decode excessive payloads"))
    monkeypatch.setattr(mido, "MidiFile", decoder)
    with pytest.raises(InvalidDataError, match="payload limit"):
        load_midi(str(path))
    decoder.assert_not_called()


def test_payload_limit_is_shared_across_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Several small metadata blocks cannot bypass the aggregate payload budget."""
    monkeypatch.setattr(midi_file, "MAX_MIDI_METADATA_BYTES", 4)
    path = tmp_path / "text.mid"
    _write_tracks(path, b"\x00\xff\x01\x03abc", b"\x00\xff\x01\x02de")
    with pytest.raises(InvalidDataError, match="payload limit"):
        load_midi(str(path))


def test_payload_at_limit_can_be_decoded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The payload limit accepts a valid boundary value, including multi-byte lengths."""
    monkeypatch.setattr(midi_file, "MAX_MIDI_METADATA_BYTES", 128)
    path = tmp_path / "boundary.mid"
    _write_tracks(
        path,
        b"\x00\xff\x01\x81\x00" + b"a" * 128 + b"\x00\x90\x3c\x64\x00\xff\x2f\x00",
    )
    assert len(load_midi(str(path)).events) == 1


def test_file_size_limit_precedes_decoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The byte limit applies to direct parser calls as well as catalog setup."""
    monkeypatch.setattr(midi_file, "MAX_MIDI_FILE_BYTES", 32)
    path = tmp_path / "large.mid"
    path.write_bytes(b"x" * 33)
    decoder = MagicMock(side_effect=AssertionError("Mido must not decode an oversized file"))
    monkeypatch.setattr(mido, "MidiFile", decoder)
    with pytest.raises(InvalidDataError, match="no larger than"):
        load_midi(str(path))
    decoder.assert_not_called()


@pytest.mark.parametrize(
    "track",
    [
        b"\x00",
        b"\x00\x3c\x64",
        b"\x00\x90\x3c",
        b"\x00\xff",
        b"\x00\xff\x01\x05ab",
        b"\x00\xf0\x04ab",
        b"\x81\x80\x80\x80\x00\x90\x3c\x64",
    ],
)
def test_malformed_event_boundaries_fail_safely(tmp_path: Path, track: bytes) -> None:
    """Truncated messages and invalid lengths cannot escape the bounded byte scan."""
    path = tmp_path / "malformed.mid"
    _write_tracks(path, track)
    with pytest.raises(InvalidDataError):
        load_midi(str(path))


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"MThd\x00\x00\x00\x05\x00\x00\x00\x01\x01\xe0",
        b"MThd\x00\x00\x00\x20\x00\x00\x00\x01\x01\xe0",
        b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0",
        b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0MTrk\x00\x00\x00\x05\x00",
    ],
)
def test_malformed_chunk_boundaries_fail_safely(tmp_path: Path, data: bytes) -> None:
    """Declared header and track sizes cannot read beyond the bounded input."""
    path = tmp_path / "chunks.mid"
    path.write_bytes(data)
    with pytest.raises(InvalidDataError):
        load_midi(str(path))


def test_same_tick_order_and_ports_survive_track_merge(tmp_path: Path) -> None:
    """Equal-tick events retain track order, prefix scope and the original tempo timeline."""
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks = [
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500000),
                mido.Message("program_change", program=40),
                mido.MetaMessage("midi_port", port=2),
                mido.Message("note_on", note=60),
                mido.MetaMessage("set_tempo", tempo=1000000, time=480),
                mido.Message("note_off", note=60, time=480),
            ]
        ),
        mido.MidiTrack(
            [
                mido.Message("program_change", program=73),
                mido.MetaMessage("midi_port", port=3),
                mido.Message("note_on", note=67),
                mido.Message("note_off", note=67, time=480),
            ]
        ),
    ]
    path = tmp_path / "ordered.mid"
    midi.save(path)
    events = load_midi(str(path)).events
    assert [(event.kind, event.port, event.param1) for event in events[:4]] == [
        ("program_change", 2, 40),
        ("note_on", 2, 60),
        ("program_change", 3, 73),
        ("note_on", 3, 67),
    ]
    assert [(event.port, event.frame) for event in events[4:]] == [
        (3, SAMPLE_RATE // 2),
        (2, SAMPLE_RATE * 3 // 2),
    ]


def _write_tracks(path: Path, *tracks: bytes) -> None:
    """Write raw event streams without having Mido normalize their running status."""
    path.write_bytes(
        b"MThd"
        + pack(">IHHH", 6, int(len(tracks) > 1), len(tracks), 480)
        + b"".join(b"MTrk" + pack(">I", len(track)) + track for track in tracks)
    )
