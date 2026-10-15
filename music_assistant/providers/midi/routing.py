"""Stable family buses and isolated native MIDI lanes for port/channel state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from music_assistant_models.errors import InvalidDataError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .midi_file import MidiEvent


class InstrumentFamily(StrEnum):
    """The seven stereo buses, in their fixed PCM output order."""

    WOODWIND = "woodwind"
    BRASS = "brass"
    STRINGS = "strings"
    HARP = "harp"
    PERCUSSION = "percussion"
    ORGAN = "organ"
    VOICES = "voices"


FAMILIES = tuple(InstrumentFamily)


@dataclass(frozen=True)
class MidiRoute:
    """A source port/channel/family mapped to an independent native synth channel."""

    port: int
    channel: int
    family: InstrumentFamily
    synth_channel: int


def instrument_family(program: int, *, percussion: bool = False) -> InstrumentFamily:
    """
    Classify a zero-based GM program, or a channel-10 drum kit, into a family bus.

    Electronic leads/pads/FX and sound-effect programs have no orchestral family
    and are explicitly unsupported in this seven-family layout.

    :param program: GM program number, from 0 through 127.
    :param percussion: Whether this is MIDI channel 10 on its original port.
    """
    if percussion or 0 <= program <= 15 or program == 47 or 112 <= program <= 119:
        return InstrumentFamily.PERCUSSION
    if 16 <= program <= 23:
        return InstrumentFamily.ORGAN
    if 24 <= program <= 45 or 48 <= program <= 51 or program in (104, 105, 106, 107, 108, 110):
        return InstrumentFamily.STRINGS
    if program == 46:
        return InstrumentFamily.HARP
    if 52 <= program <= 54:
        return InstrumentFamily.VOICES
    if 55 <= program <= 63:
        return InstrumentFamily.BRASS
    if 64 <= program <= 79 or program in (109, 111):
        return InstrumentFamily.WOODWIND
    raise InvalidDataError(f"GM program {program + 1} has no supported orchestral family")


def build_routes(events: Iterable[MidiEvent]) -> tuple[MidiRoute, ...]:
    """
    Reserve independent native lanes for each sounding source/family combination.

    A native channel modulo seven selects its audio bus. Giving a source a lane
    per family keeps already-playing voices on their original bus when a program
    change selects a different family. All lanes of a source share subsequent
    controller, note-off and program messages.

    :param events: Events in musical playback order.
    """
    programs: dict[tuple[int, int], int] = {}
    needed: set[tuple[int, int, InstrumentFamily]] = set()
    for event in events:
        address = event.port, event.channel
        if event.kind == "reset":
            programs = {key: value for key, value in programs.items() if key[0] != event.port}
        elif event.kind == "program_change":
            programs[address] = event.param1
        elif event.kind == "note_on" and event.param2:
            family = instrument_family(programs.get(address, 0), percussion=event.channel == 9)
            needed.add((*address, family))
    routes = []
    next_lane = {family: index for index, family in enumerate(FAMILIES)}
    for port, channel, family in sorted(needed):
        lane = next_lane[family]
        if lane >= 256:
            raise InvalidDataError(
                f"Too many independent {family} parts for FluidSynth's 256 MIDI channels"
            )
        routes.append(MidiRoute(port, channel, family, lane))
        next_lane[family] += len(FAMILIES)
    return tuple(routes)
