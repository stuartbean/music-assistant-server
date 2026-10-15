"""Configure one optional AirPlay output per instrument family."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from music_assistant.models.setup_flow import SetupFlowError

from .config import family_entries

if TYPE_CHECKING:
    from music_assistant.models.setup_flow import SetupSession


async def run_setup(session: SetupSession) -> None:
    """Set up or reconfigure the ensemble's family assignments."""
    values = dict(session.context.setup_data)
    errors: dict[str, SetupFlowError] | None = None
    while True:
        values.update(
            await session.form(
                [
                    replace(entry, value=values.get(entry.key))
                    for entry in family_entries(session.mass)
                ],
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
