"""Select held MIDI notes for bounded seek reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .routing import InstrumentFamily, instrument_family

if TYPE_CHECKING:
    from .midi_file import MidiEvent


def held_note_indices(events: tuple[MidiEvent, ...], end: int) -> set[int]:
    """
    Find note-on events still held by keys or pedals before an event index.

    :param events: The merged, sample-ordered MIDI timeline.
    :param end: Exclusive event index at the beginning of the seek lead-in.
    """
    channels: dict[tuple[int, int], _Channel] = {}
    for index in range(end):
        event = events[index]
        if event.kind == "reset":
            channels = {key: state for key, state in channels.items() if key[0] != event.port}
            continue
        state = channels.setdefault((event.port, event.channel), _Channel())
        if event.kind == "program_change":
            state.program = event.param1
        elif event.kind == "note_on" and event.param2:
            family = instrument_family(state.program, percussion=event.channel == 9)
            for note in state.notes:
                if note.key == event.param1 and note.family == family:
                    note.key_down = False
            state.prune()
            state.notes.append(_Note(index, event.param1, family))
        elif event.kind in ("note_off", "note_on"):
            for note in state.notes:
                if note.key == event.param1:
                    note.key_down = False
            state.prune()
        elif event.kind == "control_change":
            state.control(event.param1, event.param2)
    return {note.index for state in channels.values() for note in state.notes}


@dataclass
class _Note:
    index: int
    key: int
    family: InstrumentFamily
    key_down: bool = True
    latched: bool = False


@dataclass
class _Channel:
    program: int = 0
    sustain: bool = False
    notes: list[_Note] = field(default_factory=list)

    def control(self, control: int, value: int) -> None:
        if control == 64:
            self.sustain = value >= 64
        elif control == 66:
            for note in self.notes:
                note.latched = value >= 64
        elif control == 120:
            self.notes.clear()
        elif control == 121:
            self.sustain = False
            for note in self.notes:
                note.latched = False
        elif control == 123:
            for note in self.notes:
                note.key_down = False
        self.prune()

    def prune(self) -> None:
        self.notes = [note for note in self.notes if note.key_down or self.sustain or note.latched]
