"""goob_api: A Python package for gooby things."""

from __future__ import annotations

import logging

from goob_ai.__version__ import __version__


logging.getLogger("asyncio").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("faker").setLevel(logging.DEBUG)
logging.getLogger("sentry_sdk").setLevel(logging.WARNING)

# SOURCE: https://github.com/RSC-NA/rsc/blob/69f8ce29a6e38a960515564bf84fbd1d809468d8/rsc/llm/create_db.py#L176
# # Disable Loggers
# logging.getLogger("chromadb").setLevel(logging.ERROR)
# logging.getLogger("httpcore").setLevel(logging.ERROR)
# logging.getLogger("httpx").setLevel(logging.ERROR)
# logging.getLogger("langchain_community").setLevel(logging.ERROR)
# logging.getLogger("MARKDOWN").setLevel(logging.ERROR)
# logging.getLogger("openai").setLevel(logging.ERROR)
# logging.getLogger("unstructured").setLevel(logging.ERROR)
# logging.getLogger("urllib3").setLevel(logging.ERROR)
