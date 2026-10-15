# MidiProvider

MidiProvider exposes one server-local Standard MIDI File as a playable MA track.
Add another instance for another file. Browse the source or search for its title;
the performance also participates in normal library track sync.

## Setup

The MA container includes **FluidSynth** and the **TimGM6mb General MIDI
SoundFont**. Add **MidiProvider** in MA's music sources and enter the absolute
path to a `.mid` file, for example `/media/midi/performance.mid`. If MA runs in a
container, the file must be mounted and readable inside that container.

The advanced **Custom General MIDI SoundFont** field is optional. Leave it empty
to use the included instruments, or supply an absolute path to your own
GM-compatible `.sf2` file. Existing custom SoundFont settings are preserved.

The provider uses FluidSynth's shared library through Python's standard `ctypes`;
it does not start a MIDI/audio driver, server, or hardware connection. MA installs
the small pinned `mido` Python dependency from the provider manifest; NumPy is
already an MA dependency.

### Local Python installations

A source checkout does not use the container's bundled system packages. Install
**libfluidsynth 2.3 or later** and a GM-compatible SF2 using the operating system's
package manager. On Debian/Ubuntu, the same packages used by the MA image are:

```sh
sudo apt-get install libfluidsynth3 timgm6mb-soundfont
```

The provider automatically uses `/usr/share/sounds/sf2/TimGM6mb.sf2`. On macOS,
install FluidSynth with your package manager and select your SF2 through the
optional custom field. A missing library or unreadable/invalid file fails setup
with an error rather than creating an unplayable source.

### Building an image containing these changes

`Dockerfile.base` installs both runtime packages and checks the shared library
and default SoundFont during the image build. Build a new base image from this
branch, then build the server against that base using `BASE_IMAGE_VERSION`.
Existing published base images do not gain these packages from a server-only
rebuild. The base-image workflow publishes images; running it is a separate
release action. No published base-image version is changed by this branch.

Use the provider's **reconfigure** flow to change either file. Reload/reconfigure
after changing a file's contents. Paths are setup data, not runtime options.
Setup is restricted to users who can manage all music sources because these are
arbitrary server-local files. No existing config keys or database schemas change.

FluidSynth is LGPL-licensed; Mido is MIT-licensed. The image includes TimGM6mb
through its distribution package, including its GPL-2 license and attribution
at `/usr/share/doc/timgm6mb-soundfont/copyright`; the GPL text is at
`/usr/share/common-licenses/GPL-2`. No MIDI music is bundled. Check the license of
any custom SoundFont or MIDI arrangement separately before redistributing it.

## Supported MIDI subset

- Standard MIDI File Type 0/1 with positive PPQN division; all tracks are merged
  stably onto one timeline using tempo changes from the merged events.
- Multiple MIDI ports, each with channels 1–16 and channel 10's GM percussion.
  A port/channel address owns its own program, bank and controller state. Tracks
  are organisational containers and never determine the instrument family.
- MIDI port prefixes are track-local and persist across later events in that
  track. At a tick with one port declaration, that declaration scopes all events
  in that track at that tick, including setup messages preceding it. This handles
  exports such as the supplied Holst score. Conflicting port declarations at the
  same track/tick are rejected explicitly rather than guessed.
- Notes, velocity (including note-on velocity zero), programs, bank selection,
  pitch bend, channel/key pressure and controllers, including sustain, volume,
  expression and pan. FluidSynth handles the controller/program state.
- Bank selection uses FluidSynth's **MMA** interpretation: bank = CC0 × 128 + CC32.
  Available instruments/banks depend on the supplied SoundFont.
- GM1 System On SysEx is accepted. Other SysEx (including GS/XG/GM2 extensions),
  SMPTE division/offset, Type 2, sequencer-specific metadata,
  and omni/mono/poly channel-mode changes are rejected explicitly. Unsupported
  files fail validation before playback. Files larger than 16 MiB are rejected.

Each event is scheduled at its absolute rounded sample frame using exact rational
accumulation of MIDI ticks and tempo. Native voice changes have FluidSynth's
internal render quantum (normally 64 frames, about 1.45 ms at 44.1 kHz); there are
no wall-clock sleeps or separate MIDI playback clocks. At end-of-track, sustain
and sostenuto are released and all outstanding notes receive note-off. A fixed
two-second release tail is rendered, then EOF. Longer SoundFont releases are
truncated at that documented bound. Track duration includes the tail, rounded up
to whole seconds for MA; `session.total_frames` is the precise length.

Seeking is supported by synthesizing from frame zero and discarding the prefix
on the same worker. This restores notes, sustain, programs, controllers and voice
envelopes instead of replaying only the latest program numbers. Seeking far into
a long piece can therefore take time. MA can reuse its stereo buffer for a seek,
or refill that buffer from an earlier file position; the actual generator seek
argument determines the render session's source origin.

## One renderer and two views

The seven fixed stereo buses, in PCM order, are:

| Bus | Family | Included sounds |
|---|---|---|
| 1 | Woodwind | Saxes, reeds, pipes, bagpipe, shanai |
| 2 | Brass | Trumpet, trombone, tuba, horns, brass ensembles, orchestra hit |
| 3 | Strings | Bowed/plucked strings, ensembles, guitars, basses and ethnic plucked strings |
| 4 | Harp | Orchestral harp |
| 5 | Percussion | Drum kits, tuned percussion, timpani, piano and other struck-key instruments |
| 6 | Organ | Organs, accordion, harmonica and other GM reed-keyboard programs |
| 7 | Voices | Choir aahs, voice oohs and synth voice |

Classification uses the zero-based GM program number, with MIDI channel 10 on
**every port** always routed to percussion. GM electronic leads/pads/FX (programs
81–104 in one-based numbering) and sound effects (121–128) have no family in this
orchestral layout and are rejected if notes use them. Bank variations retain the
base GM program's family. `routing.instrument_family()` is the single mapping.

Each source-generator invocation owns one FluidSynth instance on one serial
worker thread. Internally, each source port/channel gets an independent native
MIDI lane for each family in which it plays notes. A native channel index modulo
seven selects its family output: `synth.audio-channels = synth.audio-groups = 7`.
All parts of a family mix inside the synth's one `fluid_synth_process()` call.

A program change routes **new notes** to their new family; held, sustained and
releasing voices remain on their original family bus. Controllers, bank/program
updates, note-offs, pitch bend and pressure reach every lane of that source
port/channel, keeping its parts under one control state. GM reset affects only
its source port. The native MIDI channel type is set explicitly, so reassigning
an internal lane never accidentally makes a melodic instrument a drum kit.

FluidSynth permits at most 256 native MIDI channels. This modulo routing admits
36 or 37 independent source/family combinations per bus; exceeding that bound
fails validation before playback. Native polyphony is fixed at 1,024 voices.
The Holst file fits: 69 source/family lanes across six ports, including 22
woodwind, 16 brass, 11 strings, 2 harp, 12 percussion, 1 organ and 5 voice parts.
Silent buses still occupy their fixed PCM positions.

Both shared reverb and chorus are disabled: their effect returns would otherwise
mix family content. See FluidSynth's
[rendering contract](https://www.fluidsynth.org/api/group__audio__rendering.html)
and [synth settings](https://www.fluidsynth.org/api/settings_synth.html).

The internal format is **44,100 Hz, interleaved IEEE-754 float32, little endian,
14 PCM channels (seven stereo pairs), 56 bytes/frame**. `session.format.family_names`
and `channel_layout` identify the buses: woodwind_L, woodwind_R, brass_L, brass_R,
strings_L, strings_R, harp_L, harp_R, percussion_L, percussion_R, organ_L, organ_R,
voices_L, voices_R.

Synth gain is fixed at 0.2. The ordinary MA output is stereo float32 at the same
rate: for each side, sum its seven family buses in float64, multiply by **1/16**,
clamp to [-1, 1], and convert to little-endian float32. The divisor remains a
conservative fixed headroom policy; it is not the number of active buses or parts.
This can sound quiet for sparse scores. Family buses preserve float headroom and
can exceed [-1, 1]; a future integer-PCM speaker output must apply its own gain
policy. There is no normalization, lookahead limiter, or timing shift inside the
provider. `mix_stereo()` defines the exact conversion from family buses to stereo.

## Internal Python session API

This is an in-process interface, not an HTTP/API endpoint. The exact
`StreamDetails.data` holds a typed `MidiPlaybackRequest`; MA passes those details
to `get_audio_stream()`. Its request/render ID is a UUID, not the provider track ID.
Resolving stream details reserves an ID but creates no synth or retained audio.

A future player can discover the association using its **PlayerMedia** identifiers:

```python
session = provider.resolve_playback(media.source_id, media.queue_item_id, media.queue_session_id)
consumer = session.attach(start_frame=desired_file_frame)
try:
    while block := await consumer.read(4096):
        # block.start_frame, block.frame_count, block.data
        # Route the desired instrument-family pairs from these exact file frames.
        ...
finally:
    consumer.detach()
```

The caller must resolve the provider instance that issued the queue item's stream
details (not a provider domain or library item ID). `resolve_playback()` rejects
mismatched queue generations. For an already-resolved queue item,
`provider.get_playback_session(queue_item.streamdetails)` obtains the same session
without opening another generator. Distinct requests for the same track have
independent IDs, synths and cursors. Reopening an ended/cancelled source generator
gets a new render ID, including after pause/resume. MA can stop a paused queue
after 30 seconds; preserving a live generator across pause is not assumed.

Frame zero is the original file's musical origin. A source seek to S seconds
starts published blocks at S × 44100; it does not reset frame numbering to zero.
For a pending session, `first_frame` is provisional until the actual generator
seek is known. An attachment without an explicit frame follows that origin;
an explicit file-frame cursor stays at the requested position. A standard stereo
interval at relative frame K corresponds to file frame `session.first_frame + K`
**before MA's downstream processing**. Chunk boundaries carry no timing meaning.

### Attachment and bounded retention

- Attach **before stereo consumption** to receive the whole requested interval.
  Reading the sidecar never starts or advances synthesis. Only pulling the MA
  stereo generator renders another block.
- While consumers are attached, retain at most **88,200 multichannel frames**
  (two seconds, about 4.7 MiB), shared by at most four consumers per session.
  Without consumers, retain zero multichannel frames. There are at most 16
  registered pending/active sessions per provider instance.
- Default attachment starts at the current write head. An explicit attachment
  must be inside the retained interval; a discarded prefix raises
  `FramesUnavailableError`. No implicit second render or recovery from disk occurs.
- `read(N)` waits for N frames (N must fit in the retention capacity), returning
  `PCMBlock(start_frame, frame_count, data)`. At normal EOF it returns a short
  final block if needed, then `None`. Reads are sequential per consumer; multiple
  consumers have independent cursors. Cancellation leaves the cursor unchanged.
- A slow reader that falls outside the bounded window is detached and receives
  `FramesUnavailableError`; ordinary playback continues. Detach/context exit wakes
  blocked reads and releases retention when the last consumer leaves.
- Stop, source cancellation, explicit session close and provider unload wake readers
  with `SessionClosedError`, clear retained audio and close the synth. Rendering
  errors fail stereo too; sidecar failure chains the original error. Native work
  and deletion are serialized even if cancellation arrives during synthesis.
- At EOF, native resources close immediately. Attached readers can drain the
  bounded final history, then must detach; closing the session also frees it.

### Remaining future-player integration work

MA's normal `AudioBuffer` can render/preload seconds ahead of a player's
`play_media()` and can reuse stereo audio after a render has already ended. Session
discovery remains exact, but **late attachment cannot recover discarded stems**.
The future player must arrange attachment before buffer production, actively drain
its subscription, and control its own bounded transport queues. Merely attaching
in `play_media()` does not guarantee that the intro is still available. No new
core hook is introduced in this provider milestone.

The future player must also control/account for buffer seek offsets, sample-rate
conversion, sample-format conversion, channel conversion, crossfade/overlap,
playback speed (`atempo`), fade-in, any enabled silence trimming, normalization,
DSP/filter latency and player buffering. `audio_buffer.py` can start the provider
at an earlier position than the requested queue seek; `audio.py` applies queue
processing and builds flow streams. PCM bytes do not transport these block IDs,
and MA/FFmpeg do not preserve provider chunk boundaries. This provider establishes
source-frame/session identity, not a guarantee of downstream sample identity.

## Verification

Run `pytest tests/providers/midi`. Parser, lifecycle and association tests use
small generated fixtures and a deterministic fake renderer. Native tests additionally
need libfluidsynth and a GM SoundFont; set `MIDI_TEST_SOUNDFONT=/path/to/font.sf2`
if it is outside the common system locations. Their skip reason is explicit when
these dependencies are unavailable. Native checks cover real stereo generation,
family isolation, independent port/channel state, program changes with held notes,
port-scoped resets, percussion, sustain across a seek, and exact
stereo/internal interval mixing. Actual listening on a configured MA player is a
separate verification step.
