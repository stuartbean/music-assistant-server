"""Parse Standard MIDI Files into one tempo-aware sample timeline."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from heapq import merge
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import mido
from music_assistant_models.errors import InvalidDataError

from .constants import (
    MAX_MIDI_EVENTS,
    MAX_MIDI_FILE_BYTES,
    MAX_MIDI_METADATA_BYTES,
    RELEASE_SECONDS,
    SAMPLE_RATE,
)
from .routing import MidiRoute, build_routes

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True, slots=True)
class MidiEvent:
    """A MIDI channel message scheduled at an absolute file sample frame."""

    frame: int
    kind: str
    channel: int = 0
    param1: int = 0
    param2: int = 0
    port: int = 0


@dataclass(frozen=True)
class MidiFile:
    """Validated MIDI events and their common file timeline, including a release tail."""

    name: str
    events: tuple[MidiEvent, ...]
    end_frame: int
    total_frames: int
    routes: tuple[MidiRoute, ...] = ()


def load_midi(path: str) -> MidiFile:
    """
    Load a Type 0/1 PPQN MIDI file, rejecting unsupported playback instructions.

    :param path: Local Standard MIDI File to read.
    """
    try:
        data = _read_midi_bytes(Path(path))
        _check_parse_limits(data)
        midi = mido.MidiFile(file=BytesIO(data), clip=False)
        return _timeline(midi, Path(path).stem)
    except (OSError, ValueError, EOFError, KeyError, IndexError) as err:
        raise InvalidDataError(f"Cannot play MIDI file: {err}") from err


def _read_midi_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise ValueError("Configure a readable MIDI file")
    with path.open("rb") as source:
        data = source.read(MAX_MIDI_FILE_BYTES + 1)
    if len(data) > MAX_MIDI_FILE_BYTES:
        raise ValueError("Configure a readable MIDI file no larger than 16 MiB")
    return data


def _timeline(midi: mido.MidiFile, fallback_name: str) -> MidiFile:
    """Build a stable merged timeline with exact accumulated tempo arithmetic."""
    if midi.type not in (0, 1):
        raise ValueError("Only Standard MIDI File Type 0 and Type 1 are supported")
    if midi.ticks_per_beat <= 0:
        raise ValueError("SMPTE time division is not supported; use PPQN timing")
    if midi.type == 0 and len(midi.tracks) != 1:
        raise ValueError("A Type 0 MIDI file must contain exactly one track")
    tempo = 500000
    time_us = Fraction(0)
    frame = 0
    name = fallback_name
    events: list[MidiEvent] = []
    previous_tick = 0
    for tick, port, message in _port_events(midi):
        time_us += Fraction((tick - previous_tick) * tempo, midi.ticks_per_beat)
        previous_tick = tick
        frame = round(time_us * SAMPLE_RATE / 1000000)
        kind = message.type
        if message.is_meta:
            if kind == "set_tempo":
                tempo = int(message.tempo)
                if tempo <= 0:
                    raise ValueError("MIDI tempo must be positive")
            elif kind == "track_name" and name == fallback_name and message.name.strip():
                name = message.name.strip()
            elif kind in ("smpte_offset", "sequencer_specific", "unknown_meta"):
                raise ValueError(f"Unsupported MIDI metadata: {kind}")
            continue
        if kind == "sysex":
            data = tuple(message.data)
            if len(data) == 4 and data[0] == 0x7E and data[2:] == (9, 1):
                events.append(MidiEvent(frame, "reset", port=port))
                continue
            raise ValueError("Only GM1 System On SysEx is supported (no GS/XG/GM2 extensions)")
        if kind in ("note_on", "note_off", "polytouch"):
            events.append(
                MidiEvent(
                    frame,
                    kind,
                    int(message.channel),
                    int(message.note),
                    int(message.value if kind == "polytouch" else message.velocity),
                    port=port,
                )
            )
        elif kind == "control_change":
            if message.control >= 124:
                raise ValueError("Omni/mono/poly channel-mode changes are not supported")
            events.append(
                MidiEvent(
                    frame,
                    kind,
                    int(message.channel),
                    int(message.control),
                    int(message.value),
                    port=port,
                )
            )
        elif kind in ("program_change", "pitchwheel", "aftertouch"):
            value = {"program_change": "program", "pitchwheel": "pitch", "aftertouch": "value"}
            events.append(
                MidiEvent(
                    frame, kind, int(message.channel), int(getattr(message, value[kind])), port=port
                )
            )
        else:
            raise ValueError(f"Unsupported MIDI message: {kind}")
    if not any(event.kind == "note_on" and event.param2 for event in events):
        raise ValueError("The MIDI file contains no notes")
    return MidiFile(
        name, tuple(events), frame, frame + RELEASE_SECONDS * SAMPLE_RATE, build_routes(events)
    )


def _port_events(midi: mido.MidiFile) -> Iterator[tuple[int, int, mido.Message | mido.MetaMessage]]:
    """Merge tracks while keeping port prefixes local to each track and event position."""
    yield from merge(*(_track_events(track) for track in midi.tracks), key=lambda event: event[0])


def _track_events(
    track: mido.MidiTrack,
) -> Iterator[tuple[int, int, mido.Message | mido.MetaMessage]]:
    tick = port = index = 0
    while index < len(track):
        tick += int(track[index].time)
        end = index + 1
        while end < len(track) and track[end].time == 0:
            end += 1
        # Exporters can place setup messages before their port prefix at the
        # same tick. A single declaration applies to that whole zero-duration
        # group. Conflicting declarations cannot be unambiguously repaired.
        # Scan the interval in place to avoid copying same-tick messages.
        ports = {
            int(track[position].port)
            for position in range(index, end)
            if track[position].type == "midi_port"
        }
        if len(ports) > 1:
            raise ValueError("Conflicting MIDI port declarations at the same track tick")
        if ports:
            port = ports.pop()
        while index < end:
            yield tick, port, track[index]
            index += 1


def _check_parse_limits(data: bytes) -> None:
    """Reject excessive event counts and payloads before Mido allocates decoded messages."""
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError("Invalid MIDI file header")
    header_size = int.from_bytes(data[4:8])
    if header_size < 6 or 8 + header_size > len(data):
        raise ValueError("Invalid MIDI header size")
    offset = 8 + header_size
    events = payload_bytes = 0
    for _ in range(int.from_bytes(data[10:12])):
        if data[offset : offset + 4] != b"MTrk" or offset + 8 > len(data):
            raise ValueError("Invalid MIDI track header")
        end = offset + 8 + int.from_bytes(data[offset + 4 : offset + 8])
        if end > len(data):
            raise ValueError("Truncated MIDI track")
        position = offset + 8
        running_status = 0
        while position < end:
            _delta, position = _variable_int(data, position, end)
            events += 1
            if events > MAX_MIDI_EVENTS:
                raise ValueError(f"MIDI file exceeds the {MAX_MIDI_EVENTS:,}-event limit")
            if position >= end:
                raise ValueError("Truncated MIDI event")
            status = data[position]
            if status < 0x80:
                status = running_status
            else:
                position += 1
                if status != 0xFF:
                    running_status = status
            if status in (0xF0, 0xF7, 0xFF):
                if status == 0xFF:
                    position += 1
                size, position = _variable_int(data, position, end)
                payload_bytes += size
                if payload_bytes > MAX_MIDI_METADATA_BYTES:
                    raise ValueError("MIDI metadata and SysEx exceed the 1 MiB payload limit")
            elif 0x80 <= status <= 0xEF:
                size = 1 if status >> 4 in (0xC, 0xD) else 2
            else:
                raise ValueError("Invalid or unsupported MIDI status")
            position += size
            if position > end:
                raise ValueError("Truncated MIDI event")
        offset = end


def _variable_int(data: bytes, position: int, end: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if position >= end:
            raise ValueError("Truncated MIDI variable-length value")
        byte = data[position]
        position += 1
        value = (value << 7) | (byte & 0x7F)
        if byte < 0x80:
            return value, position
    raise ValueError("MIDI variable-length values must fit in four bytes")
