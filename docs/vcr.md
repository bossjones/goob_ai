# pytest-recording: A pytest plugin that allows you recording of network interactions via VCR.py


# Getting Started

## Termonology

> Before working with pytest-recording, it is important to understand that it is built on top of VCR.py. Below are the terms used in this document, with links to more information.

- **Cassette**: A recording of network interactions.
- **VCR**: A library that records and replays network interactions.
- **VCR.py**: A Python library that provides a VCR-like interface to network interactions. Learn more [here](https://github.com/kevin1024/vcrpy). Originally inspired by Ruby's [VCR](https://github.com/vcr/vcr).
- **pytest-recording**: A pytest plugin that provides a VCR-like interface to network interactions. Learn more [here](https://github.com/pytest-dev/pytest-recording). It is an alternative to pytest-vcr.
- **Recording**: The process of capturing network interactions and saving them as a cassette.
- **Replaying**: The process of playing back network interactions from a cassette.
- **Stubbing**: The process of replacing a live URL with a local file or directory of files.
- **Mode**: The behavior of VCR.py when recording and replaying network interactions. Learn more [here](#record-modes).


## Set up credentials

An OpenAI, and Claude API key is required to make live calls to the LLM, or to run tests
without vcr (see **Running tests** section).

## Record Modes

VCR supports 4 record modes (with the same behavior as Ruby's VCR):

### once

- Replay previously recorded interactions.
- Record new interactions if there is no cassette file.
- Cause an error to be raised for new requests if there is a cassette file.

It is similar to the new_episodes record mode, but will prevent new, unexpected requests from being made (e.g. because the request URI changed).

once is the default record mode, used when you do not set one.

### new_episodes

- Record new interactions.
- Replay previously recorded interactions. It is similar to the once record mode, but will always record new interactions, even if you have an existing recorded one that is similar, but not identical.

This was the default behavior in versions < 0.3.0

### none

- Replay previously recorded interactions.
- Cause an error to be raised for any new requests. This is useful when your code makes potentially dangerous HTTP requests. The none record mode guarantees that no new HTTP requests will be made.

### all

- Record new interactions.
- Never replay previously recorded interactions. This can be temporarily used to force VCR to re-record a cassette (i.e. to ensure the responses are not out of date) or can be used when you simply want to log all HTTP requests.


## Additional Recording Options

pytest-recording provides additional options/features to record and replay network interactions. specifically:
- Straightforward pytest.mark.vcr, that reflects VCR.use_cassettes API;
- Combining multiple VCR cassettes;
- Network access blocking;
- The rewrite recording mode that rewrites cassettes from scratch.


## Example step by step guide to recording

1. Run tests without vcr (i.e. without recording) to establish baseline and make sure your tests are working.
2. Run tests with vcr `--record-mode=all` (i.e. recording) to record responses. For tests that you want to record responses for, use `@pytest.mark.vcr()`. Learn more [here](https://github.com/pytest-dev/pytest-recording) or by looking through existing tests that have cassettes recorded. `NOTE: during this recording time, your test MAY fail if the response has not been recorded yet. If so, wait until the recording is complete and re-run the test.`
3. You should now have a cassette directory in `tests` directory similar to: `/Users/malcolm/dev/malcolm/ada-agent/app/test/surfaces/slack/cassettes`. Example of what a cassette file looks like [here](/Users/malcolm/dev/malcolm/ada-agent/app/test/surfaces/slack/cassettes/slack_api_test.yaml).
4. Rerun tests with vcr `--record-mode=none` to replay the cassettes. This is now the default in make local-unittests.

# Troubleshooting NOTES:

1. PLEASE NOTE, this has not been ported over to docker yet, so you may need to run it locally till then.
2. If you get an error with a message like `VCR.Errors.CannotOverwriteExistingCassette: A cassette for this request already exists and VCR cannot automatically determine how to handle that. Please delete the existing cassette or set the `record_mode` to `all` or `new_episodes`.` then you need to re-run the tests with `--record-mode=all` to overwrite the existing cassette.
3. If you get an error with a message like `VCR.Errors.UnhandledHTTPRequestError: Unhandled HTTP request: GET http://example.com/`. This can happen if the request is not recorded in the cassette. You can fix this by:
    - deleting the cassette file and re-running the tests with `--record-mode=all` to record the response.
    - adding a `@pytest.mark.vcr()` to the test that is failing.
    - adding a `@pytest.mark.vcr(record_mode='all')` to the test that is failing.
