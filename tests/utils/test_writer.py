# pylint: disable=assigning-non-slot
# pylint: disable=consider-using-from-import
from __future__ import annotations

import logging

from typing import TYPE_CHECKING

from goob_ai.agent import AiAgent
from loguru import logger as LOGGER

import pytest


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture

import asyncio

import aiofiles

from goob_ai.utils.writer import write_file

import pytest


# Parametrized test for happy path scenarios
@pytest.mark.parametrize(
    "test_id, fname, body, filetype, directory, expected_filename",
    [
        ("HP-1", "testfile", b"Hello, world!", "txt", "/tmp", "/tmp/testfile.txt"),
        ("HP-2", "image", b"\x89PNG\r\n\x1a\n", "png", "/var/tmp", "/var/tmp/image.png"),
        ("HP-3", "data", b"1234567890", "bin", "/home/user", "/home/user/data.bin"),
    ],
)
async def test_write_file_happy_path(
    test_id,
    fname,
    body,
    filetype,
    directory,
    expected_filename,
    monkeypatch: MonkeyPatch,
    mocker: MockerFixture,
    request: FixtureRequest,
):
    mock_data = mocker.AsyncMock()
    mock_data.read.return_value = body
    # Arrange
    mock_open = mocker.patch("aiofiles.open", new_callable=mocker.MagicMock)

    mock_file = mocker.MagicMock()
    mock_open.return_value.__aenter__.return_value = mock_data

    # Act
    result = await write_file(fname, body, filetype, directory)

    # Assert
    mock_open.assert_called_once_with(expected_filename, mode="wb+")
    assert result == expected_filename


# # Parametrized test for various error cases
# @pytest.mark.parametrize(
#     "test_id, fname, body, filetype, directory, exception, expected_log",
#     [
#         (
#             "EC-1",
#             "testfile",
#             b"Hello, world!",
#             "txt",
#             "/nonexistent",
#             PermissionError,
#             "Unexpected error while writing from `testfile`: ",
#         ),
#         (
#             "EC-2",
#             "testfile",
#             b"Hello, world!",
#             "txt",
#             "/tmp",
#             OSError,
#             "Unexpected error while writing from `testfile`: ",
#         ),
#     ],
# )
# async def test_write_file_error_cases(
#     test_id: str,
#     fname: str,
#     body: str,
#     filetype: bytes | str,
#     directory: str,
#     exception: PermissionError | OSError,
#     expected_log: str,
#     caplog: pytest.LogCaptureFixture,
#     monkeypatch: MonkeyPatch,
#     mocker: MockerFixture,
#     request: FixtureRequest,
# ):
#     caplog.set_level(logging.DEBUG)
#     with caplog.at_level(logging.DEBUG):
#         # Arrange
#         mock_open = mocker.patch("aiofiles.open", side_effect=exception)
#         #
#         result = await write_file(fname, body, filetype, directory)

#         # Assert
#         assert "Unexpected error" in caplog.text
#         assert expected_log in caplog.text
#         assert result == f"{directory}/{fname}.{filetype}"
