"""Native FluidSynth verification: isolation, MIDI controls, stereo mixing and seek state."""

from pathlib import Path

import numpy as np

from music_assistant.providers.midi.constants import MIX_GAIN, PCM_CHANNELS, SAMPLE_RATE
from music_assistant.providers.midi.midi_file import MidiEvent, load_midi
from music_assistant.providers.midi.routing import FAMILIES, InstrumentFamily, MidiRoute
from music_assistant.providers.midi.session import MidiSession
from music_assistant.providers.midi.synthesizer import FluidSynth, MidiRenderer, mix_stereo
from tests.providers.midi.test_provider import make_provider


def test_native_channel_isolation_and_percussion(soundfont: str) -> None:
    """Melodic programs and GM percussion render only into their assigned stereo pairs."""
    synth = FluidSynth(soundfont)
    try:
        synth.configure_channels(
            (
                MidiRoute(0, 2, InstrumentFamily.STRINGS, 2),
                MidiRoute(0, 9, InstrumentFamily.PERCUSSION, 4),
            )
        )
        for event in (
            MidiEvent(0, "program_change", 2, 40),
            MidiEvent(0, "control_change", 2, 0, 0),
            MidiEvent(0, "control_change", 2, 32, 0),
            MidiEvent(0, "control_change", 2, 7, 100),
            MidiEvent(0, "control_change", 2, 11, 100),
            MidiEvent(0, "control_change", 2, 10, 0),
            MidiEvent(0, "pitchwheel", 2, 1000),
            MidiEvent(0, "note_on", 2, 60, 100),
            MidiEvent(0, "note_on", 4, 36, 100),
        ):
            synth.dispatch(event)
        pcm = synth.render(8192)
        assert pcm.shape == (8192, PCM_CHANNELS)
        assert np.isfinite(pcm).all()
        assert np.max(np.abs(pcm[:, 4:6])) > 0.001
        assert np.max(np.abs(pcm[:, 8:10])) > 0.001
        silent = [index for index in range(PCM_CHANNELS) if index not in (4, 5, 8, 9)]
        assert np.count_nonzero(pcm[:, silent]) == 0
        assert np.max(np.abs(pcm[:, 4])) > 20 * np.max(np.abs(pcm[:, 5]))
    finally:
        synth.close()
        synth.close()


def test_native_program_changes_affect_sound(soundfont: str) -> None:
    """The same pitch with different programs produces different rendered samples."""
    outputs = []
    for program in (0, 40):
        synth = FluidSynth(soundfont)
        try:
            synth.dispatch(MidiEvent(0, "program_change", 0, program))
            synth.dispatch(MidiEvent(0, "note_on", 0, 60, 100))
            outputs.append(synth.render(8192))
        finally:
            synth.close()
    assert not np.allclose(outputs[0], outputs[1])


async def test_real_provider_stereo_matches_internal_frames(
    midi_path: Path, soundfont: str
) -> None:
    """The public MA generator and sidecar expose two views of exactly one native rendering pass."""
    provider = make_provider(midi_path, soundfont)
    await provider.handle_async_init()
    details = await provider.get_stream_details("performance")
    session = provider.get_playback_session(details)
    consumer = session.attach(0)
    frame = 0
    peak = 0.0
    async for stereo in provider.get_audio_stream(details):
        frames = len(stereo) // 8
        block = await consumer.read(frames)
        assert block is not None
        assert block.start_frame == frame
        assert block.frame_count == frames
        pcm = np.frombuffer(block.data, dtype="<f4").reshape(-1, PCM_CHANNELS)
        assert mix_stereo(pcm) == stereo
        mix = np.frombuffer(stereo, dtype="<f4")
        assert np.isfinite(mix).all()
        assert np.max(np.abs(mix)) <= 1
        peak = max(peak, float(np.max(np.abs(mix))))
        frame += frames
    assert peak > 0.001
    assert frame == provider._midi.total_frames
    assert session.state == "eof"
    assert session._renderer is None
    assert await consumer.read() is None
    consumer.detach()
    assert not provider._sessions
    await provider.unload()


async def test_seek_preserves_sustained_note_and_controller_state(
    midi_path: Path, soundfont: str
) -> None:
    """Render-forward seeking reproduces a note held by sustain across the seek point."""
    midi = load_midi(str(midi_path))
    reference = MidiRenderer(midi, soundfont)
    try:
        prefix = reference.render(SAMPLE_RATE)
        expected = reference.render(4096)
    finally:
        reference.close()
    assert np.count_nonzero(prefix[:, 4:6]) > 0
    assert np.max(np.abs(expected[:, 4:6])) > 0.001
    session = MidiSession("seek", midi, soundfont, SAMPLE_RATE)
    consumer = session.attach(SAMPLE_RATE)
    source = session.stereo_stream()
    await anext(source)
    block = await consumer.read()
    assert block is not None
    assert block.start_frame == SAMPLE_RATE
    actual = np.frombuffer(block.data, dtype="<f4").reshape(-1, PCM_CHANNELS)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    await source.aclose()


def test_mix_has_fixed_headroom_and_no_timeline_delay() -> None:
    """The mix averages pairs, preserves frame count and clips only out-of-range sums."""
    pcm = np.ones((7, PCM_CHANNELS), dtype=np.float32)
    result = np.frombuffer(mix_stereo(pcm), dtype="<f4").reshape(-1, 2)
    np.testing.assert_array_equal(result, np.full((7, 2), len(FAMILIES) * MIX_GAIN))
    pcm *= 10
    np.testing.assert_array_equal(np.frombuffer(mix_stereo(pcm), dtype="<f4"), np.ones(14))
