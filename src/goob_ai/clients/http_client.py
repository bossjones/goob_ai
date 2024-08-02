"""http_client module containing synchronous and asynchronous HTTP clients."""

from __future__ import annotations

import http.client as http_client
import logging
import sys
import typing

from typing import Any, Dict, Optional, Union

import aiohttp
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


class AsyncHttpClient(BaseModel):
    """
    A common asynchronous HTTP client to be used by all tools.

    Sends x-request-id header with value from request_id_contextvar.
    """

    _session: aiohttp.ClientSession | None = None

    def __init__(self):
        """Initialize the AsyncHttpClient."""
        super().__init__()
        self._session = aiohttp.ClientSession()

    async def post(self, url: str, data: Any) -> aiohttp.ClientResponse:
        """
        Perform an asynchronous HTTP POST request.

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
            async with self._session.post(url, data=data, headers=self.__get_headers()) as resp:
                LOGGER.info(f"Received response code {resp.status} from url {url}")
                resp.raise_for_status()
                return resp
        except aiohttp.ClientProxyConnectionError as e:
            LOGGER.error(f"Proxy Error connecting url: {url}")
            raise e
        except aiohttp.ClientConnectionError as e:
            LOGGER.error(f"Unable to connect to url: {url}")
            raise e
        except aiohttp.ClientResponseError as e:
            LOGGER.error(f"An error occurred, method: POST, response status: {e.status}, url: {url}, Exception: {e}")
            raise e
        except aiohttp.ClientError as e:
            LOGGER.error(f"An error occurred, method: POST, url: {url}, Exception: {e}")
            raise e

    async def get(
        self, url: str, urlparams: dict[str, str] | None = None, headers: Optional[dict[str, str]] = None
    ) -> aiohttp.ClientResponse:
        """
        Perform an asynchronous HTTP GET request.

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
            async with self._session.get(
                url=url, params=urlparams, headers=self.__get_headers(additional_headers=headers)
            ) as resp:
                LOGGER.info(f"Received response code {resp.status} from url {url}")
                resp.raise_for_status()
                return resp
        except aiohttp.ClientProxyConnectionError as e:
            LOGGER.error(f"Proxy Error connecting url: {url}")
            raise e
        except aiohttp.ClientConnectionError as e:
            LOGGER.error(f"Unable to connect to url: {url}")
            raise e
        except aiohttp.ClientResponseError as e:
            LOGGER.error(f"An error occurred, method: GET, response status: {e.status}, url: {url}, Exception: {e}")
            raise e
        except aiohttp.ClientError as e:
            LOGGER.error(f"An error occurred, method: GET, url: {url}, Exception: {e}")
            raise e

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


# class AsyncClient:
#     """
#     Asynchronous Akismet API client.

#     All methods of the Akismet 1.1 and 1.2 web APIs are implemented here:

#     * :meth:`comment_check`
#     * :meth:`key_sites`
#     * :meth:`submit_ham`
#     * :meth:`submit_spam`
#     * :meth:`usage_limit`
#     * :meth:`verify_key`

#     Use of this client requires an Akismet API key; see <https://akismet.com> for
#     instructions on how to obtain one. Once you have an Akismet API key and
#     corresponding registered site URL to use with it, you can create an API client in
#     either of two ways.

#     **Recommended for most uses:** Place your Akismet API key and site URL in the
#     environment variables ``PYTHON_AKISMET_API_KEY`` and ``PYTHON_AKISMET_BLOG_URL``,
#     and then use the :meth:`validated_client` constructor:

#     .. code-block:: python

#        import akismet
#        akismet_client = await akismet.AsyncClient.validated_client()

#     This will automatically read the API key and site URL from the environment
#     variables, instantiate a client, and use its :meth:`verify_key` method to ensure the
#     key and URL are valid before returning the client instance to you. See :ref:`the FAQ
#     <alt-constructor>` for the technical reasons why the default constructor does not
#     have this behavior.

#     If you don't want to or can't use the environment variables to configure Akismet,
#     you can also explicitly configure by creating a :class:`~akismet.Config` instance
#     with your API key and site URL, and passing it as the constructor argument
#     ``config``:

#     .. code-block:: python

#        import akismet
#        config = akismet.Config(key=your_api_key, url=your_site_url)
#        akismet_client = await akismet.AsyncClient.validated_client(config=config)

#     If you rely on environment variable configuration and the complete configuration
#     cannot be found in the environment variables, :meth:`validated_client` will raise
#     :exc:`~akismet.ConfigurationError`. If the API key and URL you supply are invalid
#     according to :meth:`verify_key` -- regardless of whether you provided them via
#     environment variables or an explicit :class:`~akismet.Config` --
#     :meth:`validated_client` will raise :exc:`~akismet.APIKeyError`.

#     If you want to modify the HTTP request behavior -- for example, to support a
#     required HTTP proxy -- you can construct a custom ``httpx.AsyncClient`` and pass it
#     as the keyword argument ``http_client`` to either :meth:`validated_client` or the
#     default constructor. See :data:`akismet.USER_AGENT` for the default user-agent
#     string used by the Akismet API clients, and <https://www.python-httpx.org> for the
#     full documentation of the HTTPX module.

#     Note that if you only want to set a custom request timeout threshold (the default is
#     1 second), you can specify it by setting the environment variable
#     ``PYTHON_AKISMET_TIMEOUT`` to a value that can be parsed into a :class:`float` or
#     :class:`int` and represents the desired timeout in seconds.

#     You can also use this class as a context manager; when doing so, you do *not* need
#     to use the :meth:`validated_client` constructor, as the context manager can perform
#     the validation for you when entering the ``with`` block:

#     .. code-block:: python

#        import akismet

#        async with akismet.AsyncClient() as akismet_client:
#            # Use the client here. It will be automatically cleaned up when the "with"
#            # block exits.

#     **Unusual/advanced use cases:** Invoke the default constructor. It accepts the same
#     set of arguments as the :meth:`validated_client` constructor, and its behavior is
#     identical *except* for the fact that it will not automatically validate your
#     configuration, so you must remember to do so manually. You should only invoke the
#     default constructor if you are absolutely certain that you need to avoid the
#     automatic validation performed by :meth:`validated_client`.

#     .. warning:: **Consequences of invalid configurationn**

#        If you construct an Akismet API client through the default constructor and
#        provide an invalid key or URL, all operations of the Akismet web service, other
#        than key verification, will reply with an invalid-key message. This will cause
#        all client methods other than :meth:`verify_key` to raise
#        :exc:`~akismet.APIKeyError`. To avoid this situation, it is strongly recommended
#        that you call :meth:`verify_key` to validate your configuration prior to calling
#        any other methods, at which point you likely should be using
#        :meth:`validated_client` anyway.

#     :param config: An optional Akismet :class:`~akismet.Config`, consisting of an API
#        key and site URL.

#     :param http_client: An optional ``httpx`` async HTTP client instance to
#        use. Generally you should only pass this in if you need significantly customized
#        HTTP-client behavior, and if you do pass this argument you are responsible for
#        setting an appropriate ``User-Agent`` (see :data:`~akismet.USER_AGENT`), timeout,
#        and other configuration values. If all you want is to change the default timeout
#        (1 second), store the desired timeout, in seconds, as a floating-point or integer
#        value in the environment variable ``PYTHON_AKISMET_TIMEOUT``.

#     """

#     _http_client: httpx.AsyncClient
#     _config: "akismet.Config"

#     # Constructors.
#     # ----------------------------------------------------------------------------

#     def __init__(
#         self,
#         config: Optional["akismet.Config"] = None,
#         http_client: Optional[httpx.AsyncClient] = None,
#     ) -> None:
#         """
#         Default constructor.

#         You will almost always want to use :meth:`validated_client` instead.

#         """
#         self._config = config if config is not None else _common._try_discover_config()
#         self._http_client = http_client or _common._get_async_http_client()

#     @classmethod
#     async def validated_client(
#         cls,
#         config: Optional["akismet.Config"] = None,
#         http_client: Optional[httpx.AsyncClient] = None,
#     ) -> "AsyncClient":
#         """
#         Constructor of :class:`AsyncClient`.

#         This is usually preferred over the default ``AsyncClient()`` constructor,
#         because this constructor will validate the Akismet configuration (API key and
#         URL) prior to returning the client instance.

#         :param config: An optional explicit Akismet :class:`~akismet.Config`, consisting
#            of an API key and site URL; if not passed, the configuration will be read
#            from the environment variables ``PYTHON_AKISMET_API_KEY`` and
#            ``PYTHON_AKISMET_BLOG_URL``.

#         :param http_client: An optional ``httpx`` async HTTP client instance to
#            use. Generally you should only pass this in if you need significantly
#            customized HTTP-client behavior, and if you do pass this argument you are
#            responsible for setting an appropriate ``User-Agent`` (see
#            :data:`~akismet.USER_AGENT`), timeout, and other configuration values. If all
#            you want is to change the default timeout (1 second), store the desired
#            timeout, in seconds, as a floating-point or integer value in the environment
#            variable ``PYTHON_AKISMET_TIMEOUT``.

#         :raises akismet.APIKeyError: When the discovered Akismet configuration is
#            invalid according to :meth:`verify_key`.

#         :raises akismet.ConfigurationError: When the Akismet configuration is partially
#            or completely missing, or when the supplied site URL is in the wrong format
#            (does not begin with ``http://`` or ``https://``).

#         """
#         # While the synchronous version of the client could perform the config discovery
#         # and validation in __init__(), here we cannot because this client's
#         # verify_key() method is async, and its underlying HTTP client is async. So
#         # calling into them would require making __init__ into an async method, and
#         # Python does not currently allow __init__() to be usefully async. But a
#         # classmethod *can* be async, so we define and encourage the use of an
#         # alternative constructor in order to achieve API consistency.
#         instance = cls(config=config, http_client=http_client)
#         if not await instance.verify_key():
#             _common._configuration_error(instance._config)
#         return instance

#     # Async context-manager protocol.
#     # ----------------------------------------------------------------------------

#     async def __aenter__(self) -> "AsyncClient":
#         """
#         Entry method of the async context manager.

#         """
#         if not await self.verify_key():
#             _common._configuration_error(self._config)
#         return self

#     async def __aexit__(
#         self, exc_type: Type[BaseException], exc: BaseException, tb: TracebackType
#     ):
#         """
#         Exit method of the async context manager.

#         """
#         await self._http_client.aclose()

#     # Internal/helper methods.
#     # ----------------------------------------------------------------------------

#     async def _request(
#         self,
#         method: _common._REQUEST_METHODS,
#         version: str,
#         endpoint: str,
#         data: dict,
#     ) -> httpx.Response:
#         """
#         Make a request to the Akismet API and return the response.

#         :param method: The HTTP request method to use.

#         :param version: The Akismet API version to use.

#         :param endpoint: The Akismet API endpoint to post to.

#         :param data: The data to send in the request.

#         :raises akismet.RequestError: When an error occurs connecting to Akismet, or
#            when Akiset returns a non-success status code.

#         """
#         method = method.upper()
#         if method not in ("GET", "POST"):
#             raise _exceptions.AkismetError(
#                 f"Unrecognized request method attempted: {method}."
#             )
#         handler = getattr(self._http_client, method.lower())
#         request_kwarg = "data" if method == "POST" else "params"
#         try:
#             response = await handler(
#                 f"{_common._API_URL}/{version}/{endpoint}", **{request_kwarg: data}
#             )
#             response.raise_for_status()
#         except httpx.HTTPStatusError as exc:
#             raise _exceptions.RequestError(
#                 f"Akismet responded with error status: {exc.response.status_code}"
#             ) from exc
#         except httpx.TimeoutException as exc:
#             raise _exceptions.RequestError("Akismet timed out.") from exc
#         except httpx.RequestError as exc:
#             raise _exceptions.RequestError("Error making request to Akismet.") from exc
#         except Exception as exc:
#             raise _exceptions.RequestError("Error making request to Akismet.") from exc
#         # Since it's possible to construct a client without performing up-front API key
#         # validation, we have to watch out here for the possibility that we're making
#         # requests with an invalid key, and raise the appropriate exception.
#         if endpoint != _common._VERIFY_KEY and response.text == "invalid":
#             raise _exceptions.APIKeyError(
#                 "Akismet API key and/or site URL are invalid."
#             )
#         return response

#     async def _get_request(
#         self, version: str, endpoint: str, params: dict
#     ) -> httpx.Response:
#         """
#         Make a GET request to the Akismet API and return the response.

#         This method is used by most HTTP GET API calls.

#         :param version: The Akismet API version to use.

#         :param endpoint: The Akismet API endpoint to post to.

#         :param params: The querystring parameters to include in the request.

#         :raises akismet.APIKeyError: When the configured API key and/or site URL are
#            invalid.

#         """
#         return await self._request("GET", version, endpoint, params)

#     async def _post_request(
#         self, version: str, endpoint: str, user_ip: str, **kwargs: str
#     ) -> httpx.Response:
#         """
#         Make a POST request to the Akismet API and return the response.

#         This method is used by most HTTP POST API calls except key verification.

#         :param version: The Akismet API version to use.

#         :param endpoint: The Akismet API endpoint to post to.

#         :param user_ip: The IP address of the user who submitted the content.

#         :raises akismet.APIKeyError: When the configured API key and/or site URL are
#            invalid.

#         :raises akismet.UnknownArgumentError: When one or more unexpected optional
#            argument names are supplied. See `the Akismet documentation
#            <https://akismet.com/developers/comment-check/>`_ for details of supported
#            optional argument names.

#         """
#         unknown_args = [k for k in kwargs if k not in _common._OPTIONAL_KEYS]
#         if unknown_args:
#             raise _exceptions.UnknownArgumentError(
#                 f"Received unknown argument(s) for Akismet operation {endpoint}: "
#                 f"{', '.join(unknown_args)}"
#             )
#         data = {
#             "api_key": self._config.key,
#             "blog": self._config.url,
#             "user_ip": user_ip,
#             **kwargs,
#         }
#         return await self._request("POST", version, endpoint, data)
