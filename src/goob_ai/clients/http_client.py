"""http_client module containing synchronous and asynchronous HTTP clients."""

from __future__ import annotations

import http.client as http_client
import logging
import sys
import typing

from typing import Any, Dict, Optional, Union

import httpx
import requests
import tenacity

from loguru import logger as LOGGER
from pydantic import BaseModel
from requests import Response
from tenacity import retry_if_exception_type

from goob_ai import __version__
from goob_ai.aio_settings import aiosettings
from goob_ai.utils import retry


USER_AGENT = (
    f"goob-ai/{__version__} | Python/" f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
)
_METHODS = typing.Literal["GET", "POST"]  # pylint: disable=invalid-name
_TIMEOUT = 1.0


def _get_async_http_client() -> httpx.AsyncClient:
    """
    Return an asynchronous HTTP client for interacting with the Akismet API.

    """
    return httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=_TIMEOUT)


class HttpClient(BaseModel):
    """
    A common synchronous HTTP client to be used by all tools.

    Sends x-request-id header with value from request_id_contextvar.
    """

    _session: requests.Session | None = None

    def __init__(self):
        """Initialize the HttpClient."""
        super().__init__()
        self._session = requests.Session()
        if aiosettings.http_client_debug_enabled:
            # http.client is used by requests; setting debuglevel to a value
            # higher than zero will enable logs to stdout (this is a global
            # setting)
            http_client.HTTPConnection.debuglevel = 1

    @tenacity.retry(
        **retry.exponential_backoff_parameters(
            retry=(
                retry_if_exception_type(requests.exceptions.Timeout)
                | retry_if_exception_type(requests.exceptions.ProxyError)
                | retry_if_exception_type(requests.exceptions.ConnectionError)
                | retry_if_exception_type(requests.exceptions.HTTPError)
                | retry_if_exception_type(requests.exceptions.RequestException)
            )
        ),
        reraise=True,
    )
    def post(self, url: str, data: Any) -> Response:
        """
        Perform an HTTP POST request.

        Args:
        ----
            url: The URL to call.
            data: The request body data.

        Returns:
        -------
            The response object. If an exception happened, reraise it.

        """
        try:
            LOGGER.info(f"Calling post: {url}")
            resp = self._session.post(url, data=data, headers=self.__get_headers())
            # https://3.python-requests.org/user/quickstart/#errors-and-exceptions
            resp.raise_for_status()
            LOGGER.info(f"Received response code {resp.status_code} from url {url}")
        except requests.exceptions.ProxyError as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(f"Proxy Error connecting url: {url}")
            raise e
        except requests.exceptions.ConnectionError as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(f"Unable to connect to url: {url}")
            raise e
        except requests.exceptions.Timeout as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(f"Timeout Error connecting url: {url}")
            raise e
        # HTTPError is thrown by raise_for_status
        # SOURCE: https://stackoverflow.com/a/42379936/814221
        except requests.exceptions.HTTPError as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(
                f"An error occurred, method: POST, response status: {e.response.status_code}, url: {url}, Exception Type: {exc_type}, Exception: <{exc_value}>, response headers: {dict(e.response.headers)}, response body: {e.response.content}"
            )
            raise e
        except requests.exceptions.RequestException as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(
                f"An error occurred, method: POST, response status: {e.response.status_code}, url: {url}, Exception Type: {exc_type}, Exception: <{exc_value}>, response headers: {dict(e.response.headers)}, response body: {e.response.content}"
            )
            raise e
        return resp

    def get(
        self, url: str, urlparams: dict[str, str] | None = None, headers: Optional[dict[str, str]] = None
    ) -> Response:
        """
        Perform an HTTP GET request.

        Args:
        ----
            url: The URL to call.
            urlparams: The URL parameters.
            headers: The headers to be passed in the request.

        Returns:
        -------
            The response object. If an exception happened, reraise it.

        """
        try:
            LOGGER.info(f"Calling get: {url}")
            resp = self._session.get(url=url, params=urlparams, headers=self.__get_headers(additional_headers=headers))
            LOGGER.info(f"Received response code {resp.status_code} from url {url}")
            resp.raise_for_status()
        except requests.exceptions.ProxyError as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(f"Proxy Error connecting url: {url}")
            raise e
        except requests.exceptions.ConnectionError as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(f"Unable to connect to url: {url}")
            raise e
        except requests.exceptions.Timeout as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(f"Timeout Error connecting url: {url}")
            raise e
        # HTTPError is thrown by raise_for_status
        # SOURCE: https://stackoverflow.com/a/42379936/814221
        except requests.exceptions.HTTPError as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(
                f"An error occurred, method: GET, response status: {e.response.status_code}, url: {url}, Exception Type: {exc_type}, Exception: <{exc_value}>, response headers: {dict(e.response.headers)}, response body: {e.response.content}"
            )
            raise e
        except requests.exceptions.RequestException as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(
                f"An error occurred, method: GET, response status: {e.response.status_code}, url: {url}, Exception Type: {exc_type}, Exception: <{exc_value}>, response headers: {dict(e.response.headers)}, response body: {e.response.content}"
            )
            raise e
        return resp

    def __get_headers(self, additional_headers: Optional[dict[str, str]] = None) -> dict[str, str]:
        """
        Get the headers for the request.

        Args:
        ----
            additional_headers: Additional headers to be added to the request.

        Returns:
        -------
            The headers dictionary.

        """
        headers = {
            # "Content-Type": "application/json",
            "goob-ai": "true",
            # "x-request-id": REQUEST_ID_CONTEXTVAR.get(),
        }
        # Add additional headers if they are provided
        if additional_headers is not None:
            headers |= additional_headers
        return headers


class AsyncHttpxClient(BaseModel):
    """
    An asynchronous HTTP client using httpx to be used by all tools.

    Sends x-request-id header with value from request_id_contextvar.
    """

    _client: httpx.AsyncClient | None = None

    def __init__(self):
        """Initialize the AsyncHttpxClient."""
        super().__init__()
        self._client = httpx.AsyncClient()

    @tenacity.retry(
        **retry.exponential_backoff_parameters(
            retry=(
                retry_if_exception_type(httpx.TimeoutException)
                | retry_if_exception_type(httpx.ConnectError)
                | retry_if_exception_type(httpx.RequestError)
                | retry_if_exception_type(httpx.HTTPStatusError)
                | retry_if_exception_type(httpx.HTTPError)
            )
        ),
        reraise=True,
    )
    async def post(self, url: str, data: Any) -> httpx.Response:
        """
        Perform an asynchronous HTTP POST request using httpx.

        Args:
            url (str): The URL to call.
            data (Any): The request body data.

        Returns:
            httpx.Response: The response object. If an exception happened, reraise it.
        """
        try:
            LOGGER.info(f"Calling post: {url}")
            resp = await self._client.post(url, data=data, headers=self.__get_headers())
            LOGGER.info(f"Received response code {resp.status_code} from url {url}")
            resp.raise_for_status()
            return resp
        except httpx.TimeoutException as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(f"Timeout Error connecting url: {url}")
            raise e
        except httpx.ConnectError as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(f"Connect Errorconnecting url: {url}")
            raise e
        except httpx.RequestError as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(f"Request Error connecting url: {url}")
            LOGGER.error(f"An error occurred while requesting {e.request.url!r}.")
            raise e
        except httpx.HTTPStatusError as e:
            LOGGER.error(f"HTTP Status Error connecting url: {url}")
            LOGGER.error(f"Error response {e.response.status_code} while requesting {e.request.url!r}.")
            raise e
        except httpx.HTTPError as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(f"HTTP Error connecting url: {url}")
            # LOGGER.error(f"Error while requesting {e.request.url!r}.")
            raise e

    async def get(
        self, url: str, urlparams: dict[str, str] | None = None, headers: Optional[dict[str, str]] = None
    ) -> httpx.Response:
        """
        Perform an asynchronous HTTP GET request using httpx.

        Args:
            url (str): The URL to call.
            urlparams (dict[str, str] | None, optional): The URL parameters. Defaults to None.
            headers (Optional[dict[str, str]], optional): The headers to be passed in the request. Defaults to None.

        Returns:
            httpx.Response: The response object. If an exception happened, reraise it.
        """
        try:
            LOGGER.info(f"Calling get: {url}")
            if urlparams:
                resp: httpx.Response = await self._client.get(
                    url=url, params=urlparams, headers=self.__get_headers(additional_headers=headers)
                )
            else:
                resp: httpx.Response = await self._client.get(
                    url=url, headers=self.__get_headers(additional_headers=headers)
                )
            LOGGER.info(f"resp {resp.__dict__} from url {resp.url}")
            LOGGER.info(f"Received response code {resp.status_code} from url {url}")
            resp.raise_for_status()
            return resp
        except httpx.TimeoutException as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(f"Timeout Error connecting url: {url}")
            raise e
        except httpx.ConnectError as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(f"Connect Errorconnecting url: {url}")
            raise e
        except httpx.RequestError as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(f"Request Error connecting url: {url}")
            LOGGER.error(f"An error occurred while requesting {e.request.url!r}.")
            # LOGGER.error(
            #     f"An error occurred, method: GET, response status: {e.response.status_code}, url: {url}, Exception Type: {exc_type}, Exception: <{exc_value}>, response headers: {dict(e.response.headers)}, response body: {e.response.content}"
            # )
            raise e
        except httpx.HTTPStatusError as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(f"HTTP Status Error connecting url: {url}")
            LOGGER.error(f"Error response {e.response.status_code} while requesting {e.request.url!r}.")
            LOGGER.error(
                f"An error occurred, method: GET, response status: {e.response.status_code}, url: {url}, Exception Type: {exc_type}, Exception: <{exc_value}>, response headers: {dict(e.response.headers)}, response body: {e.response.content}"
            )
            raise e
        except httpx.HTTPError as e:
            exc_type, exc_value, _ = sys.exc_info()
            LOGGER.error(f"HTTP Error connecting url: {url}")
            # LOGGER.error(f"Error while requesting {e.request.url!r}.")
            # LOGGER.error(f"An error occurred, method: GET, response status: {e.response.status_code}, url: {url}, Exception: {e}")
            raise e
        # except httpx.HTTPError as e:
        #     LOGGER.error(f"An error occurred, method: GET, response status: {e.response.status_code}, url: {url}, Exception: {e}")
        #     raise e

    def __get_headers(self, additional_headers: Optional[dict[str, str]] = None) -> dict[str, str]:
        """
        Get the headers for the request.

        Args:
            additional_headers (Optional[dict[str, str]], optional): Additional headers to be added to the request. Defaults to None.

        Returns:
            dict[str, str]: The headers dictionary.
        """
        headers = {
            "goob-ai": "true",
        }
        if additional_headers is not None:
            headers |= additional_headers
        return headers
