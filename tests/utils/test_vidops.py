# mypy: disable-error-code="attr-defined"
# mypy: disable-error-code="index"
# mypy: disable-error-code="union-attr"
# mypy: disable-error-code="call-arg"
# mypy: disable-error-code="arg-type"
# pylint: disable=no-member

from __future__ import annotations

import asyncio

from pathlib import Path

from goob_ai.utils import vidops

import pytest


@pytest.fixture
def sample_video():
    return Path("tests/fixtures/song.mp4")


def test_calculate_bitrate():
    """Test the calculate_bitrate function."""
    assert vidops.calculate_bitrate(60, 10) == 1333
    assert vidops.calculate_bitrate(120, 5) == 333
    assert vidops.calculate_bitrate(30, 20) == 5333


@pytest.mark.asyncio
async def test_duration_video(sample_video, tmp_path, mocker):
    """Test the process_video function."""

    duration: float = await vidops.get_duration(sample_video)
    assert duration == 36.133333


@pytest.mark.asyncio
async def test_process_video(sample_video, tmp_path, mocker):
    """Test the process_video function."""
    # mocker.patch("goob_ai.utils.vidops._aio_run_process_and_communicate", return_value="60.0")
    # mocker.patch("goob_ai.utils.vidops.LOGGER.debug")

    await vidops.process_video(sample_video)

    vidops._aio_run_process_and_communicate.assert_called()
    vidops.LOGGER.debug.assert_called()


@pytest.mark.asyncio
async def test_process_audio(sample_video, tmp_path, mocker):
    """Test the process_audio function."""
    mocker.patch("goob_ai.utils.vidops._aio_run_process_and_communicate", return_value="120.0")
    mocker.patch("goob_ai.utils.vidops.LOGGER.debug")

    await vidops.process_audio(sample_video)

    vidops._aio_run_process_and_communicate.assert_called()
    vidops.LOGGER.debug.assert_called()


@pytest.mark.asyncio
async def test_process_video_low_bitrate(sample_video, tmp_path, mocker):
    """Test the process_video function with a low bitrate scenario."""
    mocker.patch("goob_ai.utils.vidops._aio_run_process_and_communicate", return_value="3600.0")
    mocker.patch("goob_ai.utils.vidops.LOGGER.debug")

    await vidops.process_video(sample_video)

    vidops._aio_run_process_and_communicate.assert_called_once()
    vidops.LOGGER.debug.assert_any_call("Target bitrate is under 150kbps.")
    vidops.LOGGER.debug.assert_any_call("Unable to compress.")


@pytest.mark.asyncio
async def test_process_audio_low_bitrate(sample_video, tmp_path, mocker):
    """Test the process_audio function with a low bitrate scenario."""
    mocker.patch("goob_ai.utils.vidops._aio_run_process_and_communicate", return_value="7200.0")
    mocker.patch("goob_ai.utils.vidops.LOGGER.debug")

    await vidops.process_audio(sample_video)

    vidops._aio_run_process_and_communicate.assert_called_once()
    vidops.LOGGER.debug.assert_any_call("Target bitrate is under 32kbps.")
    vidops.LOGGER.debug.assert_any_call("Unable to compress.")
