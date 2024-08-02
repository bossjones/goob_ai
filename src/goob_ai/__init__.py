"""goob_api: A Python package for gooby things."""

from __future__ import annotations

import logging

from goob_ai.__version__ import __version__


logging.getLogger("asyncio").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("faker").setLevel(logging.DEBUG)
logging.getLogger("sentry_sdk").setLevel(logging.WARNING)
