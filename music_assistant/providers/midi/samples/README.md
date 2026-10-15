# Bundled MIDI samples

Every `.mid` file in this directory is listed by MidiProvider in Browse and
Search, even when no custom file is configured. Files ship inside the MA package;
no download or path configuration is needed. Catalog IDs use `sample_<filename
without extension>` and must remain stable after release.

- `holst_planets.mid`: **The Planets, Op. 32**, composed by Gustav Holst.
  This is the MIDI file supplied by the developer for this implementation. The
  file contains no embedded arranger, source URL or license notice; none has
  been inferred from the composition. Keep any supplied provenance or attribution
  alongside the sample when it becomes available.

Additional samples must use the provider's supported MIDI subset. Include their
source and attribution when adding them. These are MIDI event files, not cached
or exported recordings from a music service.
