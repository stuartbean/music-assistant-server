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

- `BRIDGERTON_MAIN_THEME.mid`: **Bridgerton Main Theme**, composition credited to
  **Kris Bowers and Michael Dean Parsons**. See the
  [Television Academy's main-title theme credits](https://www.televisionacademy.com/awards/nominees-winners/2021/outstanding-original-main-title-theme-music).
  The developer supplied this MIDI arrangement for testing. It contains four
  string parts (Violin I, Violin II, Viola and Violoncello), all routed to Strings,
  and plays for about 48 seconds including the release tail. No arranger,
  arrangement source URL or license notice is embedded; the composition credits
  do not identify the arranger or establish redistribution rights for this file.
  Its stable catalog ID is `sample_BRIDGERTON_MAIN_THEME`.

Additional samples must use the provider's supported MIDI subset. Include their
source and attribution when adding them. These are MIDI event files, not cached
or exported recordings from a music service.
