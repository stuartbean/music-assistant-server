"""In-memory FluidSynth rendering with seven stereo instrument-family buses."""

from __future__ import annotations

import ctypes
from ctypes.util import find_library
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
from music_assistant_models.errors import AudioError

from .constants import BLOCK_FRAMES, MIDI_CHANNELS, MIX_GAIN, PCM_CHANNELS, SAMPLE_RATE
from .routing import FAMILIES, MidiRoute, build_routes, instrument_family

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .midi_file import MidiEvent, MidiFile


class FluidSynth:
    """An offline FluidSynth instance; use and close on a single synthesis thread."""

    def __init__(self, soundfont: str, midi_channels: int = MIDI_CHANNELS) -> None:
        """Allocate an offline synth and load its General MIDI SoundFont."""
        self.midi_channels = midi_channels
        self._settings: int | None = None
        self._synth: int | None = None
        try:
            self._lib = ctypes.CDLL(find_library("fluidsynth") or "libfluidsynth.so.3")
            self._bind()
            self._settings = self._lib.new_fluid_settings()
            self._require_settings()
            self._initialize(soundfont)
        except (OSError, AttributeError) as err:
            self.close()
            raise AudioError("Install libfluidsynth 2.3+ and a General MIDI SF2 SoundFont") from err
        except BaseException:
            self.close()
            raise

    def dispatch(self, event: MidiEvent) -> None:
        """Apply a scheduled MIDI message without advancing the audio timeline."""
        channel, first, second = event.channel, event.param1, event.param2
        if event.kind == "reset":
            self._check(self._lib.fluid_synth_system_reset(self._synth))
        elif event.kind == "note_on":
            if second:
                self._check(self._lib.fluid_synth_noteon(self._synth, channel, first, second))
            else:
                self._lib.fluid_synth_noteoff(self._synth, channel, first)
        elif event.kind == "note_off":
            # A repeated note-off (or one following all-notes-off) is valid MIDI.
            self._lib.fluid_synth_noteoff(self._synth, channel, first)
        elif event.kind == "control_change":
            self._check(self._lib.fluid_synth_cc(self._synth, channel, first, second))
        elif event.kind == "program_change":
            self._check(self._lib.fluid_synth_program_change(self._synth, channel, first))
        elif event.kind == "pitchwheel":
            self._check(self._lib.fluid_synth_pitch_bend(self._synth, channel, first + 8192))
        elif event.kind == "aftertouch":
            self._check(self._lib.fluid_synth_channel_pressure(self._synth, channel, first))
        elif event.kind == "polytouch":
            self._check(self._lib.fluid_synth_key_pressure(self._synth, channel, first, second))

    def configure_channels(self, routes: tuple[MidiRoute, ...]) -> None:
        """Initialize each native lane with its source channel's melodic/percussion type."""
        for route in routes:
            self.reset_channel(route.synth_channel, percussion=route.channel == 9)

    def reset_channel(self, channel: int, *, percussion: bool = False) -> None:
        """Reset a native channel without disturbing the other MIDI ports."""
        self._check(self._lib.fluid_synth_all_sounds_off(self._synth, channel))
        self._check(self._lib.fluid_synth_set_channel_type(self._synth, channel, int(percussion)))
        self._check(self._lib.fluid_synth_cc(self._synth, channel, 121, 0))
        for control, value in (
            (7, 100),
            (10, 64),
            (11, 127),
            (64, 0),
            (66, 0),
            (101, 0),
            (100, 0),
            (6, 2),
            (38, 0),
            (101, 127),
            (100, 127),
        ):
            self._check(self._lib.fluid_synth_cc(self._synth, channel, control, value))
        self._check(self._lib.fluid_synth_pitch_bend(self._synth, channel, 8192))
        self._check(self._lib.fluid_synth_channel_pressure(self._synth, channel, 0))
        self._check(
            self._lib.fluid_synth_bank_select(self._synth, channel, 128 if percussion else 0)
        )
        self._check(self._lib.fluid_synth_program_change(self._synth, channel, 0))

    def render(self, frames: int) -> NDArray[np.float32]:
        """Advance once and return interleaved dry audio in the fixed seven-family bus order."""
        planar = np.zeros((PCM_CHANNELS, frames), dtype=np.float32)
        pointers = (ctypes.POINTER(ctypes.c_float) * PCM_CHANNELS)(
            *(plane.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for plane in planar)
        )
        self._check(
            self._lib.fluid_synth_process(self._synth, frames, 0, None, PCM_CHANNELS, pointers)
        )
        return planar.T.copy()

    def release(self) -> None:
        """Release sustain and outstanding notes at the MIDI end-of-track instant."""
        for channel in range(self.midi_channels):
            self._check(self._lib.fluid_synth_cc(self._synth, channel, 64, 0))
            self._check(self._lib.fluid_synth_cc(self._synth, channel, 66, 0))
            self._check(self._lib.fluid_synth_all_notes_off(self._synth, channel))

    def close(self) -> None:
        """Release native resources; repeated calls are harmless."""
        if self._synth:
            self._lib.delete_fluid_synth(self._synth)
            self._synth = None
        if self._settings:
            self._lib.delete_fluid_settings(self._settings)
            self._settings = None

    def _require_settings(self) -> None:
        if not self._settings:
            raise AudioError("Could not allocate FluidSynth settings")

    def _initialize(self, soundfont: str) -> None:
        for key, value in (
            ("synth.midi-channels", self.midi_channels),
            ("synth.audio-channels", len(FAMILIES)),
            ("synth.audio-groups", len(FAMILIES)),
            ("synth.reverb.active", 0),
            ("synth.chorus.active", 0),
            ("synth.cpu-cores", 1),
            ("synth.polyphony", 1024),
            ("synth.min-note-length", 0),
        ):
            if self._lib.fluid_settings_setint(self._settings, key.encode(), value) != 0:
                raise AudioError(f"FluidSynth does not support setting {key}")
        for key, number in (("synth.sample-rate", SAMPLE_RATE), ("synth.gain", 0.2)):
            if self._lib.fluid_settings_setnum(self._settings, key.encode(), number) != 0:
                raise AudioError(f"FluidSynth does not support setting {key}")
        if self._lib.fluid_settings_setstr(self._settings, b"synth.midi-bank-select", b"mma") != 0:
            raise AudioError("FluidSynth does not support MMA bank selection")
        self._synth = self._lib.new_fluid_synth(self._settings)
        if not self._synth:
            raise AudioError("Could not create FluidSynth")
        if self._lib.fluid_synth_sfload(self._synth, soundfont.encode(), 1) < 0:
            raise AudioError("Could not load the configured General MIDI SoundFont")
        self._check(self._lib.fluid_synth_set_channel_type(self._synth, 9, 1))

    def _bind(self) -> None:
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        signatures: dict[str, tuple[Any, list[Any]]] = {
            "new_fluid_settings": (pointer, []),
            "delete_fluid_settings": (None, [pointer]),
            "fluid_settings_setint": (integer, [pointer, ctypes.c_char_p, integer]),
            "fluid_settings_setnum": (integer, [pointer, ctypes.c_char_p, ctypes.c_double]),
            "fluid_settings_setstr": (integer, [pointer, ctypes.c_char_p, ctypes.c_char_p]),
            "new_fluid_synth": (pointer, [pointer]),
            "delete_fluid_synth": (None, [pointer]),
            "fluid_synth_sfload": (integer, [pointer, ctypes.c_char_p, integer]),
            "fluid_synth_process": (
                integer,
                [pointer, integer, integer, pointer, integer, pointer],
            ),
        }
        for name, count in (
            ("set_channel_type", 2),
            ("noteon", 3),
            ("noteoff", 2),
            ("cc", 3),
            ("program_change", 2),
            ("pitch_bend", 2),
            ("channel_pressure", 2),
            ("key_pressure", 3),
            ("system_reset", 0),
            ("all_notes_off", 1),
            ("all_sounds_off", 1),
            ("bank_select", 2),
        ):
            signatures[f"fluid_synth_{name}"] = (integer, [pointer, *([integer] * count)])
        for name, (result, arguments) in signatures.items():
            function = getattr(self._lib, name)
            function.restype = result
            function.argtypes = arguments

    def _check(self, result: int) -> None:
        if result < 0:
            raise AudioError("FluidSynth could not render the requested MIDI program or event")


class MidiRenderer:
    """Render a parsed file in bounded blocks, dispatching events on its sample timeline."""

    def __init__(self, midi: MidiFile, soundfont: str) -> None:
        """Start a fresh rendering pass at file frame zero."""
        self._midi = midi
        self.routes = midi.routes or build_routes(midi.events)
        lanes = max((route.synth_channel for route in self.routes), default=0) + 1
        native_channels = max(MIDI_CHANNELS, ((lanes + 15) // 16) * 16)
        self._synth = FluidSynth(soundfont, native_channels)
        try:
            self._synth.configure_channels(self.routes)
        except BaseException:
            self._synth.close()
            raise
        self._programs: dict[tuple[int, int], int] = {}
        self._source_routes: dict[tuple[int, int], list[MidiRoute]] = {}
        for route in self.routes:
            self._source_routes.setdefault((route.port, route.channel), []).append(route)
        self.frame = 0
        self._event = 0
        self._released = False

    def render(self, frames: int = BLOCK_FRAMES) -> NDArray[np.float32]:
        """Return at most the requested frames, or an empty array at end-of-stream."""
        end = min(self.frame + frames, self._midi.total_frames)
        result = np.empty((end - self.frame, PCM_CHANNELS), dtype=np.float32)
        offset = 0
        while self.frame < end:
            while self._event < len(self._midi.events):
                event = self._midi.events[self._event]
                if event.frame > self.frame:
                    break
                self._dispatch(event)
                self._event += 1
            if self.frame >= self._midi.end_frame and not self._released:
                self._synth.release()
                self._released = True
            boundary = end
            if self._event < len(self._midi.events):
                boundary = min(boundary, self._midi.events[self._event].frame)
            if not self._released:
                boundary = min(boundary, self._midi.end_frame)
            count = boundary - self.frame
            result[offset : offset + count] = self._synth.render(count)
            offset += count
            self.frame = boundary
        return result

    def close(self) -> None:
        """Release the renderer and its SoundFont."""
        self._synth.close()

    def _dispatch(self, event: MidiEvent) -> None:
        if event.kind == "reset":
            for route in self.routes:
                if route.port == event.port:
                    self._synth.reset_channel(route.synth_channel, percussion=route.channel == 9)
                    self._programs.pop((route.port, route.channel), None)
            return
        address = event.port, event.channel
        if event.kind == "program_change":
            self._programs[address] = event.param1
        routes = self._source_routes.get(address, ())
        if event.kind == "note_on" and event.param2:
            family = instrument_family(
                self._programs.get(address, 0), percussion=event.channel == 9
            )
            routes = [route for route in routes if route.family == family]
        # Notes start only on their selected family lane. Later controller and
        # note-off events reach every lane so held/releasing notes remain in the
        # family where they started, even across a program change.
        for route in routes:
            self._synth.dispatch(replace(event, channel=route.synth_channel))


def mix_stereo(pcm: NDArray[np.float32]) -> bytes:
    """Mix the seven stereo family buses at fixed 1/16 gain into clipped little-endian float32 stereo."""
    mixed = pcm.reshape(-1, len(FAMILIES), 2).sum(axis=1, dtype=np.float64) * MIX_GAIN
    return bytes(np.clip(mixed, -1.0, 1.0).astype("<f4").tobytes())
