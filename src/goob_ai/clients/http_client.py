"""http_client"""

from __future__ import annotations

import http.client as http_client
import logging
import sys

from typing import Any, Dict, Optional, Union

import requests
import tenacity

from loguru import logger as LOGGER
from pydantic import BaseModel
from requests import Response
from tenacity import retry_if_exception_type

from goob_ai.aio_settings import aiosettings
from goob_ai.utils import retry


class HttpClient(BaseModel):
    """
    A common http client to be used by all tools.
    Sends x-request-id header with value from request_id_contextvar.
    """

    _session: requests.Session | None = None

    def __init__(self):
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
        Perform an http post request
        :param url: the url to call
        :param data: request body data
        :return: the response object. If an exception happened, reraise it.
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
        Perform an http get request
        :param url: the url to call
        :param urlparams: url parameters
        :param headers: headers to be passed in the request
        :return: the response object. If an exception happened, reraise it.
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
        headers = {
            # "Content-Type": "application/json",
            "goob-ai": "true",
            # "x-request-id": REQUEST_ID_CONTEXTVAR.get(),
        }
        # Add additional headers if they are provided
        if additional_headers is not None:
            headers |= additional_headers
        return headers
