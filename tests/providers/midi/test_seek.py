"""Held-note selection for fast seeks across MIDI ports, pedals and families."""

import pytest

from music_assistant.providers.midi.midi_file import MidiEvent
from music_assistant.providers.midi.seek import held_note_indices


def note(key: int = 60, *, channel: int = 0, port: int = 0) -> MidiEvent:
    """Create a note-on for a held-note reconstruction fixture."""
    return MidiEvent(0, "note_on", channel, key, 100, port)


def cc(control: int, value: int) -> MidiEvent:
    """Create a channel-zero controller event."""
    return MidiEvent(0, "control_change", 0, control, value)


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ((note(),), {0}),
        ((note(), MidiEvent(0, "note_off", 0, 60)), set()),
        ((note(), MidiEvent(0, "note_on", 0, 60, 0)), set()),
        ((cc(64, 127), note(), MidiEvent(0, "note_off", 0, 60)), {1}),
        ((cc(64, 127), note(), MidiEvent(0, "note_off", 0, 60), cc(64, 0)), set()),
        ((note(), cc(66, 127), MidiEvent(0, "note_off", 0, 60)), {0}),
        ((cc(66, 127), note(), MidiEvent(0, "note_off", 0, 60)), set()),
        ((note(), cc(66, 127), MidiEvent(0, "note_off", 0, 60), cc(66, 0)), set()),
        ((cc(64, 127), note(), cc(123, 0)), {1}),
        ((note(), cc(123, 0)), set()),
        ((cc(64, 127), note(), cc(120, 0)), set()),
        ((cc(64, 127), note(), MidiEvent(0, "note_off", 0, 60), cc(121, 0)), set()),
        ((note(), cc(121, 0)), {0}),
        ((note(), note()), {1}),
        ((cc(64, 127), note(), note()), {1, 2}),
        ((note(), MidiEvent(0, "program_change", 0, 56), note()), {0, 2}),
        ((note(), MidiEvent(0, "program_change", 0, 56), note(), cc(123, 0)), set()),
        ((note(), note(port=1), MidiEvent(0, "reset", port=0)), {1}),
        ((note(), note(channel=1), MidiEvent(0, "note_off", 0, 60)), {1}),
    ],
)
def test_held_notes(events: tuple[MidiEvent, ...], expected: set[int]) -> None:
    """Reconstruction retains only notes held at the exact prefix boundary."""
    assert held_note_indices(events, len(events)) == expected


def test_later_pedal_release_does_not_change_prefix() -> None:
    """Events in the lead-in remain scheduled normally, including the boundary itself."""
    events = (cc(64, 127), note(), MidiEvent(0, "note_off", 0, 60), cc(64, 0))
    assert held_note_indices(events, 3) == {1}
