"""Collect server-local MIDI and General MIDI SoundFont paths."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from music_assistant.models.setup_flow import SetupFlowError

from .constants import CONFIG_ENTRIES

if TYPE_CHECKING:
    from music_assistant.models.setup_flow import SetupSession


async def run_setup(session: SetupSession) -> None:
    """Configure or reconfigure the files used by MidiProvider."""
    values = dict(session.context.setup_data)
    errors: dict[str, SetupFlowError] | None = None
    while True:
        values.update(
            await session.form(
                [replace(entry, value=values.get(entry.key)) for entry in CONFIG_ENTRIES],
                step_id="user",
                errors=errors,
                last_step=True,
            )
        )
        try:
            await session.finish(values)
            return
        except SetupFlowError as err:
            errors = {"base": err}
