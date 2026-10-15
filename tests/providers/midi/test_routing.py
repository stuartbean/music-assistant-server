"""Port/channel state and stable instrument-family output routing."""

from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
import pytest
from music_assistant_models.errors import InvalidDataError

from music_assistant.providers.midi.constants import PCM_CHANNELS, SAMPLE_RATE
from music_assistant.providers.midi.midi_file import MidiEvent, MidiFile, load_midi
from music_assistant.providers.midi.routing import (
    InstrumentFamily,
    build_routes,
    instrument_family,
)
from music_assistant.providers.midi.session import PCMFormat
from music_assistant.providers.midi.synthesizer import MidiRenderer


@pytest.mark.parametrize(
    ("program", "family"),
    [
        (73, InstrumentFamily.WOODWIND),
        (60, InstrumentFamily.BRASS),
        (40, InstrumentFamily.STRINGS),
        (46, InstrumentFamily.HARP),
        (11, InstrumentFamily.PERCUSSION),
        (19, InstrumentFamily.ORGAN),
        (52, InstrumentFamily.VOICES),
        (0, InstrumentFamily.PERCUSSION),
    ],
)
def test_program_family(program: int, family: InstrumentFamily) -> None:
    """GM programs map deterministically to the documented family buses."""
    assert instrument_family(program) == family
    assert instrument_family(program, percussion=True) == InstrumentFamily.PERCUSSION


def test_family_layout_is_stable() -> None:
    """Every block has seven stereo buses even if most families are silent."""
    assert PCM_CHANNELS == 14
    assert PCMFormat().channel_layout == (
        "woodwind_L",
        "woodwind_R",
        "brass_L",
        "brass_R",
        "strings_L",
        "strings_R",
        "harp_L",
        "harp_R",
        "percussion_L",
        "percussion_R",
        "organ_L",
        "organ_R",
        "voices_L",
        "voices_R",
    )
    assert PCMFormat().bytes_per_frame == 56


def test_ports_and_setup_before_prefix(tmp_path: Path) -> None:
    """Each track's same-tick port declaration also scopes its instrument setup."""
    midi = mido.MidiFile(type=1)
    for port, program in ((0, 73), (1, 60)):
        midi.tracks.append(
            mido.MidiTrack(
                [
                    mido.Message("program_change", channel=0, program=program),
                    mido.MetaMessage("midi_port", port=port),
                    mido.Message("note_on", channel=0, note=60, velocity=100),
                    mido.Message("note_off", channel=0, note=60, time=480),
                ]
            )
        )
    path = tmp_path / "ports.mid"
    midi.save(path)
    score = load_midi(str(path))
    assert {
        (event.port, event.param1) for event in score.events if event.kind == "program_change"
    } == {(0, 73), (1, 60)}
    assert {(route.port, route.channel, route.family) for route in score.routes} == {
        (0, 0, InstrumentFamily.WOODWIND),
        (1, 0, InstrumentFamily.BRASS),
    }


def test_conflicting_same_tick_port_declarations_fail(tmp_path: Path) -> None:
    """Ambiguous routing is rejected instead of silently merging unrelated parts."""
    midi = mido.MidiFile(type=0)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("midi_port", port=0),
                mido.MetaMessage("midi_port", port=1),
                mido.Message("note_on", note=60),
            ]
        )
    )
    path = tmp_path / "conflict.mid"
    midi.save(path)
    with pytest.raises(InvalidDataError, match="Conflicting MIDI port"):
        load_midi(str(path))


def test_unsupported_family_fails_explicitly() -> None:
    """Electronic sounds outside the seven-family contract are not silently mislabeled."""
    with pytest.raises(InvalidDataError, match="no supported orchestral family"):
        build_routes((MidiEvent(0, "program_change", 0, 80), MidiEvent(0, "note_on", 0, 60, 100)))


def test_native_routing_limit_fails_before_rendering() -> None:
    """An oversubscribed family fails rather than aliasing independent controller states."""
    events: list[MidiEvent] = []
    for port in range(40):
        events.extend(
            (
                MidiEvent(0, "program_change", 0, 73, port=port),
                MidiEvent(0, "note_on", 0, 60, 100, port=port),
            )
        )
    with pytest.raises(InvalidDataError, match="Too many independent woodwind"):
        build_routes(events)


def test_native_parts_on_same_family_keep_independent_volume(soundfont: str) -> None:
    """Muting one port does not mute a different part mixed into the same bus."""
    events = (
        MidiEvent(0, "program_change", 0, 73, port=0),
        MidiEvent(0, "program_change", 0, 73, port=1),
        MidiEvent(0, "control_change", 0, 7, 0, port=0),
        MidiEvent(0, "note_on", 0, 60, 100, port=0),
        MidiEvent(0, "note_on", 0, 67, 100, port=1),
    )
    midi = MidiFile("ports", events, SAMPLE_RATE, SAMPLE_RATE, build_routes(events))
    renderer = MidiRenderer(midi, soundfont)
    try:
        actual = renderer.render(4096)
    finally:
        renderer.close()
    reference_events = (events[1], events[-1])
    reference = MidiRenderer(
        MidiFile(
            "one port", reference_events, SAMPLE_RATE, SAMPLE_RATE, build_routes(reference_events)
        ),
        soundfont,
    )
    try:
        expected = reference.render(4096)
    finally:
        reference.close()
    assert np.max(np.abs(actual[:, :2])) > 0.001
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert np.count_nonzero(actual[:, 2:]) == 0


def test_held_notes_keep_family_and_controllers_follow_program_change(soundfont: str) -> None:
    """A sustained string voice survives a switch to flute; source volume still controls both."""
    events = (
        MidiEvent(0, "program_change", 0, 40),
        MidiEvent(0, "control_change", 0, 64, 127),
        MidiEvent(0, "note_on", 0, 60, 100),
        MidiEvent(SAMPLE_RATE // 4, "note_off", 0, 60),
        MidiEvent(SAMPLE_RATE // 2, "program_change", 0, 73),
        MidiEvent(SAMPLE_RATE // 2, "note_on", 0, 67, 100),
        MidiEvent(SAMPLE_RATE, "control_change", 0, 7, 0),
    )
    renderer = MidiRenderer(
        MidiFile("switch", events, 2 * SAMPLE_RATE, 2 * SAMPLE_RATE, build_routes(events)),
        soundfont,
    )
    try:
        pcm = renderer.render(2 * SAMPLE_RATE)
    finally:
        renderer.close()
    middle = pcm[3 * SAMPLE_RATE // 4 : SAMPLE_RATE]
    assert np.max(np.abs(middle[:, :2])) > 0.001
    assert np.max(np.abs(middle[:, 4:6])) > 0.001
    assert np.max(np.abs(pcm[SAMPLE_RATE + 1024 :])) < 1e-6
    assert np.count_nonzero(pcm[:, [2, 3, 6, 7, 8, 9, 10, 11, 12, 13]]) == 0


def test_gm_reset_is_scoped_to_its_port(soundfont: str) -> None:
    """A reset on one source port does not silence another port sharing the synth."""
    events = (
        MidiEvent(0, "program_change", 0, 40, port=0),
        MidiEvent(0, "program_change", 0, 73, port=1),
        MidiEvent(0, "note_on", 0, 60, 100, port=0),
        MidiEvent(0, "note_on", 0, 67, 100, port=1),
        MidiEvent(SAMPLE_RATE // 2, "reset", port=0),
    )
    renderer = MidiRenderer(
        MidiFile("reset", events, SAMPLE_RATE, SAMPLE_RATE, build_routes(events)), soundfont
    )
    try:
        pcm = renderer.render(SAMPLE_RATE)
    finally:
        renderer.close()
    after_reset = pcm[3 * SAMPLE_RATE // 4 :]
    assert np.max(np.abs(after_reset[:, :2])) > 0.001
    assert np.count_nonzero(after_reset[:, 4:6]) == 0
