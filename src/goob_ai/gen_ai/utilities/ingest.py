from __future__ import annotations

import asyncio
import os
import tempfile

from typing import TYPE_CHECKING
from urllib.parse import urljoin

import aiohttp
import pinecone

from bs4 import BeautifulSoup
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger as LOGGER
from pinecone import Pinecone


if TYPE_CHECKING:
    from goob_ai.goob_bot import AsyncGoobBot


async def download_file(bot: AsyncGoobBot, session: aiohttp.ClientSession, url: str, output_directory: str):
    """
    Downloads a file from a given URL.

    Args:
    ----
      bot (Bot): The bot instance.
      session (aiohttp.ClientSession): The aiohttp session.
      url (str): The URL of the file to download.
      output_directory (str): The directory to save the file to.
    Side Effects:
      Writes the file to the output directory.

    Examples:
    --------
      >>> download_file(bot, session, "https://example.com/file.txt", "/tmp/")

    """
    async with session.get(url) as response:
        if response.status == 200:
            file_name = os.path.join(output_directory, os.path.basename(url))
            file_content = await response.read()
            with open(file_name, "wb") as file:
                file.write(file_content)
            LOGGER.debug(f"Downloaded: {url}")
        else:
            LOGGER.error(f"Failed to download: {url}")

        await LOGGER.complete()


async def ingest(bot: AsyncGoobBot, url: str, namespace: str):
    """
    Ingests documents from a given URL into Pinecone.

    Args:
    ----
      bot (Bot): The bot instance.
      url (str): The URL of the documents to ingest.
      namespace (str): The namespace to ingest the documents into.
    Side Effects:
      Ingests documents into Pinecone.

    Examples:
    --------
      >>> ingest_db(bot, "https://example.com/db", "my_namespace")

    """
    base_url = url

    with tempfile.TemporaryDirectory() as temp_dir:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url) as response:
                if response.status == 200:
                    soup = BeautifulSoup(await response.text(), "html.parser")
                    tasks = []

                    for link in soup.find_all("a", {"class": "reference internal"}):
                        file_url = urljoin(base_url, link["href"])
                        if file_url.endswith(".html"):
                            tasks.append(download_file(bot, session, file_url, temp_dir))

                    await asyncio.gather(*tasks)
                else:
                    LOGGER.error(f"Failed to download: {base_url}")

        class MyReadThedbLoader(ReadTheDocsLoader):
            """My custom ReadThedbLoader."""

            def _clean_data(self, data: str) -> str:
                """
                Cleans the data from a given HTML string.

                Args:
                ----
                  data (str): The HTML string to clean.

                Returns:
                -------
                  str: The cleaned string.

                Examples:
                --------
                  >>> MyReadThedbLoader._clean_data("<html><body>Hello World!</body></html>")
                  'Hello World!'

                """
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(data, **self.bs_kwargs)

                html_tags = [
                    ("div", {"role": "main"}),
                    ("main", {"id": "main-content"}),
                    ("body", {}),
                ]

                text = None

                for tag, attrs in html_tags[::-1]:
                    text = soup.find(tag, attrs)
                    if text is not None:
                        break

                text = text.get_text() if text is not None else ""
                return "\n".join([t for t in text.split("\n") if t])

        loader = MyReadThedbLoader(temp_dir, features="html.parser", encoding="utf-8")
        db = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        texts = text_splitter.split_documents(db)
        print("f{texts}")
        await LOGGER.complete()
