"""Parse Standard MIDI Files into one tempo-aware sample timeline."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING

import mido
from music_assistant_models.errors import InvalidDataError

from .constants import RELEASE_SECONDS, SAMPLE_RATE
from .routing import MidiRoute, build_routes

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
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
        midi = mido.MidiFile(path, clip=False)
        return _timeline(midi, Path(path).stem)
    except (OSError, ValueError, EOFError, KeyError, IndexError) as err:
        raise InvalidDataError(f"Cannot play MIDI file: {err}") from err


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
    merged: list[tuple[int, int, int, int, mido.Message | mido.MetaMessage]] = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        absolute = []
        for message in track:
            tick += int(message.time)
            absolute.append((tick, message))
        port = 0
        for tick, group in groupby(absolute, key=lambda item: item[0]):
            messages = [item[1] for item in group]
            # Exporters can place setup messages before their port prefix at the
            # same tick. A single declaration applies to that whole zero-duration
            # group. Conflicting declarations cannot be unambiguously repaired.
            ports = {int(message.port) for message in messages if message.type == "midi_port"}
            if len(ports) > 1:
                raise ValueError("Conflicting MIDI port declarations at the same track tick")
            if ports:
                port = ports.pop()
            for index, message in enumerate(messages):
                merged.append((tick, track_index, index, port, message))
    merged.sort(key=lambda item: item[:3])
    for tick, _track, _index, port, message in merged:
        yield tick, port, message
