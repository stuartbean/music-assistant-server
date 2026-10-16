# MidiPlayer: first cut

MidiPlayer registers one **MIDI Ensemble** per provider instance. Add/configure
AirPlay and MidiProvider first, then add MidiPlayer and select one optional
AirPlay receiver for each of the seven instrument families. One receiver can
serve several families. Unassigned families are silent. Physical players remain
owned and discovered by the existing AirPlay provider; no duplicate speakers or
nested sync groups are created.

Play a MidiProvider track on MIDI Ensemble. Reconfigure MidiPlayer to change its
assignments. A receiver already in another group must be removed from that group
first. All selected receivers must be available and belong to the same AirPlay
provider instance. AirPlay protocol endpoints linked to other MA players are
listed using their parent player's display name.

## Volume

MIDI Ensemble uses MA's standard group volume and mute controls. The group slider
adjusts the assigned speakers through their configured volume controls, using
MA's existing balance interpolation. Adjust an individual speaker to change its
balance within the ensemble. The displayed group level is the highest volume of
its available, powered-on members that support volume control.

Configured receivers remain listed while idle, so group volume can be set before
playback. During playback and pause, membership follows the live AirPlay session.
Receivers are held by the ensemble only while that session exists and are released
on stop or failed startup. Linked AirPlay endpoints appear as their visible parent
players, whose configured controls handle volume and mute.

## Audio and timing

- Standard provider stereo consumption drives one MidiProvider render. Before
  consuming it, the player attaches to that exact render's family PCM stream.
- Every reference interval is matched to a sample-addressed `PCMBlock`. NumPy
  sums the assigned left/right family channels for each receiver in float64,
  applies fixed 1/16 gain, clips to [-1, 1], and emits stereo float32 at 44.1 kHz.
  Mapping all families to one receiver reproduces the standard mix.
- Only one interval's reference/mixes are retained by the bridge, alongside
  MidiProvider's bounded two-second family history. No independent renderer,
  full-piece PCM cache, sidecar file or public audio route is introduced.
- The existing `AirPlayStreamSession` receives the reference stream for its
  common frame counter, but its optional per-receiver PCM callback substitutes
  each receiver's family mix before the existing encoder. The reference mix is
  never played. Callback output must retain the exact byte/frame count.
- Cold start, shared clock selection, receiver readiness, coordinated START,
  per-receiver format conversion, flushes and warm restarts remain in AirPlay.
- The virtual player's progress reports the shared audible anchor, not the
  synthesis write head. MA adds the seek offset to this relative stream position.

## Playback lifecycle

The `Player.uses_audio_buffer` capability defaults to True. MidiPlayer returns
False: initial and next-track stereo prebuffering are skipped, and flow mode is
disabled for this direct-source player. It consumes the ordinary provider audio
interface itself, without MA's stereo-only cache/queue-processing chain. Other
players keep the existing buffering behavior.

Every play/seek reserves fresh MIDI stream details under the exact queue item and
queue generation before attachment. This also handles a previously played item
with an existing stereo buffer: the cache cannot stand in for missing family
frames. Seeks retain MidiProvider's fast state reconstruction and absolute source
frame positions. AirPlay warm-replaces the source and flushes/re-anchors all
receivers together. If that fails, a fresh render is created for cold restart.

Pause parks the AirPlay session and closes the render. Resume returns through the
queue's resume path and starts a fresh render. Stop, failed starts and unload
release the producer, consumer, synth and receivers. End-of-file drains through
the receiver pipelines before the virtual player reports idle and MA advances
its queue. Lost members follow the existing AirPlay removal/error handling.

The same 20 ms half-cosine fade is applied to every receiver after an explicit
seek. An existing three-second resume fade is also honored. Neither changes
sample counts. Outgoing audio already buffered at receivers cannot be faded
reliably by the server.

## First-cut boundaries and review

- Fixed membership during playback; stop/reconfigure/restart to change outputs.
  Late joining is rejected for receiver-specific sessions because the existing
  AirPlay late-join ring holds the reference mix, not each receiver's family mix.
- MIDI tracks only. No independent family queues, left/right reassignment,
  per-family gain, independent software master gain or mixing of arbitrary music sources.
- Normal playback speed only. Crossfades, queue normalization and other master
  queue processing are bypassed, rather than applied only to the discarded
  reference. Receiver output processing still uses AirPlay's normal output plan;
  differing receiver DSP/latency settings need accounting for in acoustic tests.
- One shared transport anchor establishes the intended timing; physical
  multi-speaker synchronization still needs measurement/listening on the actual
  receivers. AirPlay's existing per-receiver sync adjustment remains available.

Tests cover family routing/gain/fades/frame continuity, exact render association,
setup, duplicate receiver assignments, pause/stop/seek and failed-start cleanup,
warm fallback, queue buffer bypass and the unchanged ordinary AirPlay fan-out.
Run `pytest tests/providers/midi_player tests/providers/midi
 tests/providers/airplay/test_stream_session.py`. Native tests need FluidSynth and
an SF2 as documented by MidiProvider.
