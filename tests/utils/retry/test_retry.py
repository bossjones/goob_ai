"""test_retry"""
# pylint: disable=E1121
# mypy: disable-error-code="arg-type"

from __future__ import annotations
import logging
import unittest.mock
from typing import Union

import tenacity

import pytest


from goob_ai.utils import retry

MockType = Union[
    unittest.mock.MagicMock,
    unittest.mock.AsyncMock,
    unittest.mock.NonCallableMagicMock,
]

logger = logging.getLogger(__name__)


@pytest.mark.unittest
@pytest.mark.configonly
def test__base_parameters(mocker) -> None:
    """
    test the base parameters for tenacity
    """

    # INFO: https://pytest-mock.readthedocs.io/en/latest/usage.html#spys
    _spy_base_parameters: MockType = mocker.spy(retry, "_base_parameters")
    spy_retry_settings: MockType = mocker.spy(retry, "aiosettings")
    spy_tenacity_before_sleep_log: MockType = mocker.spy(retry.tenacity, "before_sleep_log")
    spy_tenacity_stop_after_attempt: MockType = mocker.spy(retry.tenacity, "stop_after_attempt")
    test_params = retry._base_parameters()

    assert isinstance(test_params, dict)
    assert "before_sleep" in test_params
    assert "stop" in test_params

    spy_retry_settings.assert_called_with()
    _spy_base_parameters.assert_called_once_with()
    spy_tenacity_stop_after_attempt.assert_called_once_with(
        spy_retry_settings.spy_return.retry_stop_after_attempt
    )  # default = 3

    # NOTE: We use mocker.ANY here because the logger instance
    # is relative to the module it was called from.
    # EXAMPLE: <Logger app.test.utilities.retry.test_retry (WARNING)> vs <Logger utilities.retry (WARNING)>
    spy_tenacity_before_sleep_log.assert_called_once_with(mocker.ANY, logging.WARN)


@pytest.mark.configonly
def test_linear_backoff_parameters(mocker) -> None:
    """
    test linear parameters for tenacity
    """

    # Use mocker.spy to inspect function calls
    _spy_base_parameters: MockType = mocker.spy(retry, "_base_parameters")
    # INFO: https://pytest-mock.readthedocs.io/en/latest/usage.html#spys
    spy_retry_settings: MockType = mocker.spy(retry, "aiosettings")
    spy_tenacity_wait_fixed: MockType = mocker.spy(retry.tenacity, "wait_fixed")
    spy_linear_backoff_parameters: MockType = mocker.spy(retry, "linear_backoff_parameters")
    test_params = retry.linear_backoff_parameters()

    assert isinstance(test_params, dict)
    assert "before_sleep" in test_params
    assert "stop" in test_params
    assert "wait" in test_params

    _spy_base_parameters.assert_called_with()
    spy_linear_backoff_parameters.assert_called_with()
    spy_retry_settings.assert_called_with()
    spy_tenacity_wait_fixed.assert_called_with(spy_retry_settings.spy_return.retry_wait_fixed)  # default = 15


@pytest.mark.configonly
def test_exponential_backoff_parameters(mocker) -> None:
    """
    test linear parameters for tenacity
    """

    # Use mocker.spy to inspect function calls
    # INFO: https://pytest-mock.readthedocs.io/en/latest/usage.html#spys
    _spy_base_parameters = mocker.spy(retry, "_base_parameters")
    spy_retry_settings = mocker.spy(retry, "aiosettings")
    spy_tenacity_wait_exponential = mocker.spy(retry.tenacity, "wait_exponential")
    spy_exponential_backoff_parameters = mocker.spy(retry, "exponential_backoff_parameters")
    test_params = retry.exponential_backoff_parameters()

    assert isinstance(test_params, dict)
    assert "before_sleep" in test_params
    assert "stop" in test_params
    assert "wait" in test_params

    _spy_base_parameters.assert_called_with()
    spy_exponential_backoff_parameters.assert_called_with()
    spy_retry_settings.assert_called_with()
    spy_tenacity_wait_exponential.assert_called_with(
        min=spy_retry_settings.spy_return.retry_wait_exponential_min,  # default = 1
        max=spy_retry_settings.spy_return.retry_wait_exponential_max,  # default = 5
        multiplier=spy_retry_settings.spy_return.retry_wait_exponential_multiplier,  # default = 2
    )


@pytest.mark.configonly
def test_is_result_none() -> None:
    """
    Function that can be assigned to the retry_if_result paramter to
    tenacity.retry to conditionally retry if None is returned.
    """

    assert not retry.is_result_none(1)


@pytest.mark.configonly
def test_return_outcome_result() -> None:
    """
    Callback that can be assigned to the retry_error_callback parameter to
    tenacity.retry to return the last return value instead of raising RetryError
    when the retry stop condition is reached.
    """

    retry_state = tenacity.RetryCallState(None, None, (), {})

    assert not retry.return_outcome_result(retry_state)
