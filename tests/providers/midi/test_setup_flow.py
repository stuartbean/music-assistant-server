"""User setup can use the included samples and instrument bank."""

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from music_assistant.models.setup_flow import SetupSession
from music_assistant.providers.midi.constants import CONF_MIDI_FILE, CONF_SOUNDFONT
from music_assistant.providers.midi.setup_flow import run_setup
from tests.providers.midi.test_provider import make_provider


async def test_setup_accepts_optional_custom_file(midi_path: Path) -> None:
    """The Add provider form can complete without selecting a SoundFont."""
    session = MagicMock()
    session.context.setup_data = {}
    session.form = AsyncMock(return_value={CONF_MIDI_FILE: str(midi_path)})
    session.finish = AsyncMock()
    await run_setup(cast("SetupSession", session))
    entries = {entry.key: entry for entry in session.form.call_args.args[0]}
    assert entries[CONF_MIDI_FILE].required is False
    assert entries[CONF_SOUNDFONT].required is False
    assert entries[CONF_SOUNDFONT].advanced
    session.finish.assert_awaited_once_with({CONF_MIDI_FILE: str(midi_path)})


@pytest.mark.parametrize("font_value", [None, ""])
async def test_missing_or_empty_soundfont_uses_included_default(
    midi_path: Path, soundfont: str, font_value: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider with no custom font loads and synthesizes using the included default."""
    provider = make_provider(midi_path)
    if font_value is None:
        provider.config.setup_data.pop(CONF_SOUNDFONT)
    else:
        provider.config.setup_data[CONF_SOUNDFONT] = font_value
    monkeypatch.setattr("music_assistant.providers.midi.provider.DEFAULT_SOUNDFONT", soundfont)
    await provider.handle_async_init()
    assert provider._soundfont == soundfont
    details = await provider.get_stream_details("performance")
    source = provider.get_audio_stream(details)
    try:
        assert len(await anext(source)) > 0
    finally:
        await source.aclose()
        await provider.unload()


async def test_existing_custom_soundfont_is_preserved(
    midi_path: Path, soundfont: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing custom path takes precedence even when the bundled font is unavailable."""
    provider = make_provider(midi_path, soundfont)
    monkeypatch.setattr(
        "music_assistant.providers.midi.provider.DEFAULT_SOUNDFONT", "/missing/default.sf2"
    )
    await provider.handle_async_init()
    assert provider._soundfont == soundfont
    await provider.unload()


async def test_setup_accepts_bundled_defaults() -> None:
    """Adding the provider can finish with no custom file or SoundFont paths."""
    session = MagicMock()
    session.context.setup_data = {}
    session.form = AsyncMock(return_value={})
    session.finish = AsyncMock()
    await run_setup(cast("SetupSession", session))
    session.finish.assert_awaited_once_with({})
