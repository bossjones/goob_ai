"""Test the async Typer class."""
from __future__ import annotations

import asyncio

import pytest

from goob_ai.utils.asynctyper import AsyncTyper


@pytest.mark.asyncio
async def test_async_command() -> None:
    """Test the async_command decorator."""
    app = AsyncTyper()

    @app.async_command()
    async def hello() -> None:
        """Say hello."""
        await asyncio.sleep(1)
        print("Hello!")

    await hello()
    assert True
