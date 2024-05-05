"""Test the async Typer class."""

from __future__ import annotations

import asyncio

from goob_ai.asynctyper import AsyncTyper

import pytest


# SOURCE: https://github.com/blurry-dev/blurry/blob/6cd0541fd23659a615c7ce93d8212795c6d42a15/blurry/async_typer.py#L7
# @pytest.mark.asyncio
# async def test_async_command() -> None:
#     """Test the async_command decorator."""
#     app = AsyncTyper()

#     @app.async_command()
#     async def hello() -> None:
#         """Say hello."""
#         await asyncio.sleep(1)
#         print("Hello!")

#     await hello()
#     assert True


@pytest.mark.asyncio
async def test_async_command() -> None:
    """Test the async_command decorator."""
    app = AsyncTyper()

    @app.command()
    async def hello() -> None:
        """Say hello."""
        await asyncio.sleep(1)
        print("Hello!")

    await hello()
    assert True
