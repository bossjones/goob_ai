# pylint: disable=assigning-non-slot
# pylint: disable=consider-using-from-import
from __future__ import annotations

import asyncio

from typing import TYPE_CHECKING, Optional

import httpx
import pytest_asyncio
import respx

from goob_ai.clients.http_client import AsyncHttpxClient
from httpx import ConnectError, HTTPError, HTTPStatusError, RequestError, TimeoutException
from loguru import logger as LOGGER
from tenacity import RetryError

import pytest


# FIXME: Turn these into pytest fixtures going forward
TEST_URL = "https://www.mydomainname.com"
TEST_RESPONSE = {
    "response": "This is the response",
}

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.mark.unittest()
@pytest.mark.httpclientonly()
@pytest.mark.asyncio()
async def test_post_success(respx_mock: respx.MockRouter, caplog: LogCaptureFixture):
    """
    Test a successful POST request.

    Args:
        respx_mock (respx.MockRouter): The respx mock router.
    """
    respx_mock.post(TEST_URL).mock(return_value=httpx.Response(200, json=TEST_RESPONSE))

    test_http_client = AsyncHttpxClient()
    async with test_http_client._client:
        response = await test_http_client.post(TEST_URL, "data")
        assert response.status_code == 200
        assert response.json() == TEST_RESPONSE


@pytest.mark.unittest()
@pytest.mark.httpclientonly()
@pytest.mark.asyncio()
async def test_post_bad_http_code(respx_mock: respx.MockRouter, caplog: LogCaptureFixture):
    """
    Test a POST request with a bad HTTP status code.

    Args:
        respx_mock (respx.MockRouter): The respx mock router.
    """
    respx_mock.post(TEST_URL).mock(return_value=httpx.Response(500, json=TEST_RESPONSE))

    test_http_client = AsyncHttpxClient()
    async with test_http_client._client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await test_http_client.post(TEST_URL, "data")
        assert exc_info.value.response.status_code == 500


@pytest.mark.unittest()
@pytest.mark.httpclientonly()
@pytest.mark.asyncio()
async def test_get_success(respx_mock: respx.MockRouter, caplog: LogCaptureFixture):
    """
    Test a successful GET request.

    Args:
        respx_mock (respx.MockRouter): The respx mock router.
    """
    respx_mock.get(TEST_URL).mock(return_value=httpx.Response(200, json=TEST_RESPONSE))

    test_http_client = AsyncHttpxClient()
    async with test_http_client._client:
        response = await test_http_client.get(TEST_URL)
        assert response.status_code == 200
        assert response.json() == TEST_RESPONSE


@pytest.mark.unittest()
@pytest.mark.httpclientonly()
@pytest.mark.asyncio()
async def test_get_bad_http_code(respx_mock: respx.MockRouter, caplog: LogCaptureFixture):
    """
    Test a GET request with a bad HTTP status code.

    Args:
        respx_mock (respx.MockRouter): The respx mock router.
    """
    respx_mock.get(TEST_URL).mock(return_value=httpx.Response(500, json=TEST_RESPONSE))

    test_http_client = AsyncHttpxClient()
    async with test_http_client._client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await test_http_client.get(TEST_URL)
        assert exc_info.value.response.status_code == 500
