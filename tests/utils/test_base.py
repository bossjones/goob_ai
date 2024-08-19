from __future__ import annotations

import logging
import os
import shutil
import sys

from collections.abc import Generator, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Set, Union

from goob_ai.utils import base
from loguru import logger as LOGGER

import pytest


if TYPE_CHECKING:
    from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock

    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


def test_get_sys_module() -> None:
    assert base.get_sys_module() is sys


def test_get_itertools_module() -> None:
    import itertools

    assert base.get_itertools_module() is itertools


def test_check_whoami(mocker: MockerFixture) -> None:
    expected_function_name = "test_check_whoami"
    expected_filename = __file__

    mock_sys = mocker.MagicMock()
    mock_sys._getframe.return_value.f_code.co_name = expected_function_name
    mock_sys._getframe.return_value.f_lineno = 42
    mock_sys._getframe.return_value.f_code.co_filename = expected_filename
    mocker.patch("goob_ai.utils.base.get_sys_module", return_value=mock_sys)

    function_name, line_number, filename = base.check_whoami()

    assert function_name == expected_function_name
    assert line_number == 42
    assert filename == expected_filename


def test_check_callersname(mocker: MockerFixture) -> None:
    expected_function_name = "test_check_callersname"
    expected_filename = __file__

    mock_sys = mocker.MagicMock()
    mock_sys._getframe.return_value.f_code.co_name = expected_function_name
    mock_sys._getframe.return_value.f_lineno = 42
    mock_sys._getframe.return_value.f_code.co_filename = expected_filename
    mocker.patch("goob_ai.utils.base.get_sys_module", return_value=mock_sys)

    function_name, line_number, filename = base.check_callersname()

    assert function_name == expected_function_name
    assert line_number == 42
    assert filename == expected_filename


@pytest.mark.utilsonly()
def test_print_line_seperator(capsys: CaptureFixture) -> None:
    base.print_line_seperator("test")
    captured = capsys.readouterr()
    assert captured.out == "----------------- test -----------------\n"

    base.print_line_seperator("test", length=20)
    captured = capsys.readouterr()
    assert captured.out == "------- test -------\n"

    base.print_line_seperator("test", char="*")
    captured = capsys.readouterr()
    assert captured.out == "***************** test *****************\n"


@pytest.mark.utilsonly()
def test_print_output(capsys: CaptureFixture) -> None:
    base.print_output("test", "output")
    captured = capsys.readouterr()
    assert captured.out == "test output\n"

    base.print_output("test", "output", sep="-")
    captured = capsys.readouterr()
    assert captured.out == "test-output\n"

    base.print_output("test", "output", end="")
    captured = capsys.readouterr()
    assert captured.out == "test output"


@pytest.mark.utilsonly()
def test_create_dict_from_filter() -> None:
    d = {"a": 1, "b": 2, "c": 3}
    white_list = ["a", "c"]
    result = base.create_dict_from_filter(d, white_list)
    assert result == {"a": 1, "c": 3}


@pytest.mark.utilsonly()
@pytest.mark.parametrize(
    "node, whitelist, expected",
    [
        (
            {"a": 1, "b": {"c": 2, "d": 3}, "e": [{"f": 4, "g": 5}, {"h": 6}]},
            ["a", "c", "f"],
            {"a": 1, "b": {"c": 2}, "e": [{"f": 4}]},
        ),
        ({"a": 1, "b": 2}, ["c"], None),
        ([{"a": 1, "b": 2}, {"c": 3}], ["a"], [{"a": 1}]),
        ([], ["a"], None),
    ],
)
def test_fltr(node: Union[dict, list], whitelist: list[str], expected: Union[dict, list, None]) -> None:
    assert base.fltr(node, whitelist) == expected


@pytest.mark.utilsonly()
def test_dict_merge() -> None:
    dct = {"a": 1, "b": {"c": 2}}
    merge_dct = {"b": {"d": 3}, "e": 4}
    expected = {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}
    assert base.dict_merge(dct, merge_dct) == expected

    dct = {"a": 1, "b": {"c": 2}}
    merge_dct = {"b": {"d": 3}, "e": 4}
    expected = {"a": 1, "b": {"c": 2}}
    assert base.dict_merge(dct, merge_dct, add_keys=False) == expected
