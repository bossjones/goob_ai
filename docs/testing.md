# pytest-recording

## Record Modes

VCR supports 4 record modes (with the same behavior as Ruby's VCR):

### once

- Replay previously recorded interactions.
- Record new interactions if there is no cassette file.
- Cause an error to be raised for new requests if there is a cassette file.

It is similar to the new_episodes record mode, but will prevent new, unexpected requests from being made (e.g. because
the request URI changed).

once is the default record mode, used when you do not set one.

### new_episodes

- Record new interactions.
- Replay previously recorded interactions. It is similar to the once record mode, but will always record new
    interactions, even if you have an existing recorded one that is similar, but not identical.

This was the default behavior in versions \< 0.3.0

### none

- Replay previously recorded interactions.
- Cause an error to be raised for any new requests. This is useful when your code makes potentially dangerous HTTP
    requests. The none record mode guarantees that no new HTTP requests will be made.

### all

- Record new interactions.
- Never replay previously recorded interactions. This can be temporarily used to force VCR to re-record a cassette (i.e.
    to ensure the responses are not out of date) or can be used when you simply want to log all HTTP requests.
