"""Generate dspy.Signatures"""

from __future__ import annotations
import typer
import asyncio

from goob_ai.asynctyper import AsyncTyper

# app = typer.Typer(help="dummy command")
app = AsyncTyper(help="dummy command")


@app.command("dummy")
def cli_dummy_cmd(prompt: str):
    """Generate a new dspy.Module. Example: dspygen sig new 'text -> summary'"""
    return f"dummy cmd: {prompt}"


@app.cli_dummy_cmd()
async def aio_cli_dummy_cmd() -> str:
    """Returns information about the bot."""
    await asyncio.sleep(1)
    return "slept for 1 second"
    # typer.echo("This is GoobBot CLI")


if __name__ == "__main__":
    app()
