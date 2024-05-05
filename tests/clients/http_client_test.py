from __future__ import annotations
import requests
from requests import (
    HTTPError,
    RequestException,
)
from requests_mock.mocker import Mocker
from tenacity import RetryError

# Pytest
import pytest


from goob_ai.clients.http_client import HttpClient

# FIXME: Turn these into pytest fixtures going forward
TEST_URL = "https://www.mydomainname.com"
TEST_RESPONSE = {
    "response": "This is the response",
}


@pytest.mark.unittest
@pytest.mark.httpclientonly
def test_post_success(requests_mock: Mocker):
    requests_mock.post(
        TEST_URL,
        json={
            "response": TEST_RESPONSE,
        },
    )

    test_http_client = HttpClient()
    response = test_http_client.post(TEST_URL, "data")
    assert response.status_code == 200


@pytest.mark.unittest
@pytest.mark.httpclientonly
def test_post_bad_http_code(requests_mock: Mocker):
    requests_mock.post(
        TEST_URL,
        json={
            "response": TEST_RESPONSE,
        },
        status_code=500,
    )

    test_http_client = HttpClient()
    # tenacity goes through a number of different exceptions before it gets to the final one
    with pytest.raises((Exception, RetryError, HTTPError, RequestException)) as e:
        response = test_http_client.post(TEST_URL, "data")
        assert response.status_code == 500


# FIXME: Now that we have a better understanding of what is possible with 'requests_mock',
# we can add more tests to cover other scenarios in a followup PR. For now this should be sufficent.
@pytest.mark.unittest
@pytest.mark.httpclientonly
def test_post_throws(mocker):
    err_str = "The server encountered an internal error"
    mocked_response = mocker.create_autospec(requests.Response)
    mocked_response.status_code = 500
    mocked_response.url = TEST_URL
    mocked_response.reason = "Internal Server Error"
    mocked_response.method = "POST"
    mocked_response.ok.return_value = False
    mocked_response.text.return_value = err_str
    mocked_response.body = err_str

    _mock_client_post = mocker.patch("goob_ai.clients.http_client.HttpClient.post")
    _mock_client_post.side_effect = [RequestException("Failed to connect"), mocked_response]

    test_http_client = HttpClient()

    # tenacity goes through a number of different exceptions before it gets to the final one
    with pytest.raises((Exception, RetryError, HTTPError, RequestException)) as e:
        test_http_client.post(TEST_URL, "data")
    assert "Failed to connect" in str(e.value)


def test_get_success(requests_mock: Mocker):
    requests_mock.get(
        TEST_URL,
        json={
            "response": TEST_RESPONSE,
        },
    )

    http_client = HttpClient()
    response = http_client.get(url=TEST_URL, urlparams={"key": "value"})
    assert response.status_code == 200


@pytest.mark.unittest
@pytest.mark.httpclientonly
def test_get_throws(mocker: Mocker):
    err_str = "The server encountered an internal error"
    mocked_response = mocker.create_autospec(requests.Response)
    mocked_response.status_code = 500
    mocked_response.url = TEST_URL
    mocked_response.reason = "Internal Server Error"
    mocked_response.method = "GET"
    mocked_response.ok.return_value = False
    mocked_response.text.return_value = err_str
    mocked_response.body = err_str
    _mock_client_post = mocker.patch("goob_ai.clients.http_client.HttpClient.get")
    _mock_client_post.side_effect = [RequestException("Failed to connect"), mocked_response]
    http_client = HttpClient()
    with pytest.raises((Exception, RetryError, HTTPError, RequestException)) as e:
        http_client.get(TEST_URL, "data")

    assert "Failed to connect" in str(e.value)


@pytest.mark.unittest
@pytest.mark.httpclientonly
def test_get_bad_http_code(requests_mock: Mocker):
    requests_mock.get(
        TEST_URL,
        json={
            "response": TEST_RESPONSE,
        },
        status_code=500,
    )

    test_http_client = HttpClient()
    # tenacity goes through a number of different exceptions before it gets to the final one
    with pytest.raises((Exception, RetryError, HTTPError, RequestException)) as e:
        response = test_http_client.get(TEST_URL, "data")
        assert response.status_code == 500
