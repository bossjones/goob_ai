"""goob_ai.downloader"""

# SOURCE: https://github.com/Fogapod/KiwiBot/blob/49743118661abecaab86388cb94ff8a99f9011a8/modules/utils/module_screenshot.py
# Python script containing the different algorithms used to download videos from
# various sources
from __future__ import annotations

import asyncio
import logging
import pathlib
import ssl

import aiofile
import aiohttp
import certifi


# LOGGER = get_logger(__name__, provider="Downloader", level=logging.DEBUG)
from loguru import logger as LOGGER

from goob_ai.bot_logger import get_logger


VERIFY_SSL = False


# SOURCE: https://stackoverflow.com/questions/35388332/how-to-download-images-with-aiohttp
async def download_and_save(url: str, dest_override=False):
    # SOURCE: https://github.com/aio-libs/aiohttp/issues/955
    sslcontext = ssl.create_default_context(cafile=certifi.where())
    sslcontext.check_hostname = False
    sslcontext.verify_mode = ssl.CERT_NONE
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=sslcontext if VERIFY_SSL else None)) as http:
        url_file_api = pathlib.Path(url)
        filename = dest_override or f"{url_file_api.name}"
        # breakpoint()
        async with http.request("GET", url, ssl=sslcontext if VERIFY_SSL else None) as resp:
            if resp.status == 200:
                # SOURCE: https://stackoverflow.com/questions/72006813/python-asyncio-file-write-after-request-getfile-not-working
                size = 0
                try:
                    async with aiofile.async_open(filename, "wb+") as afp:
                        async for chunk in resp.content.iter_chunked(1024 * 512):  # 500 KB
                            await afp.write(chunk)
                            size += len(chunk)
                except asyncio.TimeoutError:
                    LOGGER.error(f"A timeout ocurred while downloading '{filename}'")

                await LOGGER.complete()
                return filename, size
