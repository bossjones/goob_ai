"""Package Utilities."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import logging
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time
import traceback

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Set, Union
from uuid import uuid4

import bpdb
import bs4
import chromadb
import httpx
import pysnooper
import uritools

from chromadb.api import ClientAPI, ServerAPI
from chromadb.config import Settings as ChromaSettings
from httpx import ConnectError
from langchain.evaluation import load_evaluator
from langchain_chroma import Chroma
from langchain_chroma import Chroma as ChromaVectorStore
from langchain_community.document_loaders import (
    DirectoryLoader,
    JSONLoader,
    PyMuPDFLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, MarkdownTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger as LOGGER
from pydantic.v1.types import SecretStr
from tqdm import tqdm

from goob_ai import llm_manager, redis_memory
from goob_ai.aio_settings import aiosettings
from goob_ai.utils import file_functions


HERE = os.path.dirname(__file__)

DATA_PATH = os.path.join(HERE, "..", "data", "chroma", "documents")
CHROMA_PATH = os.path.join(HERE, "..", "data", "chroma", "vectorstorage")
CHROMA_PATH_API = Path(CHROMA_PATH)
WEBBASE_LOADER_PATTERN = r"^https?://[a-zA-Z0-9.-]+\.github\.io(/.*)?$"
EXCLUDE_KEYS_FROM_CHECKSUM = {"metadata": {"chunk_id", "id", "checksum", "last_seen_at", "item_id"}}
DAY_IN_SECONDS = 24 * 3600


def get_nested_value(d: dict, keys: str) -> str:
    """
    Extract nested value from dict.

    Example:
      >>> get_nested_value({"a": "v1", "c1": {"c2": "v2"}}, "c1.c2")
      'v2'
    """

    d = copy.deepcopy(d)
    for key in keys.split("."):
        if d and isinstance(d, dict) and d.get(key):
            d = d[key]
        else:
            return ""
    return str(d)


def stringify_dict(d: dict, keys: list[str]) -> str:
    """Stringify all values in a dictionary.

    Example:
        >>> d_ = {"a": {"text": "Apify is cool"}, "description": "Apify platform"}
        >>> stringify_dict(d_, ["a.text", "description"])
        'a.text: Apify is cool\\ndescription: Apify platform'
    """
    return "\n".join([f"{key}: {value}" for key in keys if (value := get_nested_value(d, key))])


# FIXME: wrapper function
def get_dataset_loader(filename: str) -> TextLoader | PyMuPDFLoader | WebBaseLoader | None:
    """Get the appropriate loader for the given filename.

    Args:
        filename: The name of the file.

    Returns:
        The loader for the given file type, or None if the file type is not supported.
    """
    return get_rag_loader(filename)


# SOURCE: https://github.com/RSC-NA/rsc/blob/69f8ce29a6e38a960515564bf84fbd1d809468d8/rsc/llm/create_db.py#L176
async def string_to_doc(text: str) -> Document:
    """
    Convert a string to a Document object.

    Args:
        text (str): The input string to convert.

    Returns:
        Document: The converted Document object.
    """
    return Document(page_content=text)


# SOURCE: https://github.com/RSC-NA/rsc/blob/69f8ce29a6e38a960515564bf84fbd1d809468d8/rsc/llm/create_db.py#L176
async def markdown_to_documents(docs: list[Document]) -> list[Document]:
    """
    Split Markdown documents into smaller chunks.

    Args:
        docs (list[Document]): The list of Markdown documents to split.

    Returns:
        list[Document]: The list of split document chunks.
    """
    md_splitter = MarkdownTextSplitter()
    return md_splitter.split_documents(docs)


# SOURCE: https://github.com/RSC-NA/rsc/blob/69f8ce29a6e38a960515564bf84fbd1d809468d8/rsc/llm/create_db.py#L176
def franchise_metadata(record: dict, metadata: dict) -> dict:
    """
    Update the metadata dictionary with franchise-specific information.

    Args:
        record (dict): The record dictionary.
        metadata (dict): The metadata dictionary to update.

    Returns:
        dict: The updated metadata dictionary.
    """
    LOGGER.debug(f"Metadata: {metadata}")
    metadata["source"] = "API"
    return metadata


# # SOURCE: https://github.com/RSC-NA/rsc/blob/69f8ce29a6e38a960515564bf84fbd1d809468d8/rsc/llm/create_db.py#L176
# async def json_to_docs(data: str, jq_schema: str, metadata_func: Callable | None) -> list[Document]:
#     """
#     Convert JSON data to a list of Document objects.

#     Args:
#         data (str): The JSON data as a string.
#         jq_schema (str): The jq schema to apply to the JSON data.
#         metadata_func (Callable | None): The function to apply to the metadata.

#     Returns:
#         list[Document]: The list of converted Document objects.
#     """
#     with tempfile.NamedTemporaryFile() as fd:
#         if isinstance(data, str):
#             fd.write(data.encode("utf-8"))
#         elif isinstance(data, bytes):
#             fd.write(data)
#         else:
#             raise TypeError("JSON data must be str or bytes")
#         loader = JSONLoader(
#             file_path=fd.name,
#             jq_schema=jq_schema,
#             text_content=False,
#             metadata_func=franchise_metadata,
#         )
#         chunks = loader.load()

#     return chunks


# SOURCE: https://github.com/RSC-NA/rsc/blob/69f8ce29a6e38a960515564bf84fbd1d809468d8/rsc/llm/create_db.py#L176
async def generate_document_hashes(docs: list[Document]) -> list[str]:
    """
    Generate hashes for a list of Document objects.

    Args:
        docs (list[Document]): The list of Document objects to generate hashes for.

    Returns:
        list[str]: The list of generated hashes.
    """
    hashes = []
    for doc in docs:
        source = doc.metadata.get("source")
        api_id = doc.metadata.get("id")

        if source and api_id:
            ident = f"{source}/{api_id}"
        elif source:
            ident = f"{source}"
        elif api_id:
            LOGGER.warning(f"LLM Document has no source: {doc.page_content[:50]}")
            ident = f"{api_id}"
        else:
            LOGGER.warning(f"LLM Document has no metadata: {doc.page_content[:50]}")
            ident = doc.page_content

        hash = hashlib.sha256(ident.encode("utf-8")).hexdigest()
        hashes.append(hash)

    await LOGGER.complete()
    return hashes


def get_suffix(filename: str) -> str:
    """Get the file extension from the given filename.

    Args:
        filename: The name of the file.

    Returns:
        The file extension in lowercase without the leading period.
    """
    ext = get_file_extension(filename)
    ext_without_period = remove_leading_period(ext)
    LOGGER.debug(f"ext: {ext}, ext_without_period: {ext_without_period}")
    return ext


def get_file_extension(filename: str) -> str:
    """Get the file extension from the given filename.

    Args:
        filename: The name of the file.

    Returns:
        The file extension in lowercase.
    """
    return pathlib.Path(filename).suffix.lower()


def remove_leading_period(ext: str) -> str:
    """Remove the leading period from the file extension.

    Args:
        ext: The file extension.

    Returns:
        The file extension without the leading period.
    """
    return ext.replace(".", "")


def is_pdf(filename: str) -> bool:
    """Check if the given filename has a PDF extension.

    Args:
        filename: The name of the file.

    Returns:
        True if the file has a PDF extension, False otherwise.
    """
    suffix = get_suffix(filename)
    res = suffix in file_functions.PDF_EXTENSIONS
    LOGGER.debug(f"res: {res}")
    return res


def is_txt(filename: str) -> bool:
    """Check if the given filename has a text extension.

    Args:
        filename: The name of the file.

    Returns:
        True if the file has a text extension, False otherwise.
    """
    suffix = get_suffix(filename)
    res = suffix in file_functions.TXT_EXTENSIONS
    LOGGER.debug(f"res: {res}")
    return res


def is_valid_uri(uri: str) -> bool:
    """
    Check if the given URI is valid.

    Args:
        uri (str): The URI to check.

    Returns:
        bool: True if the URI is valid, False otherwise.
    """
    parts = uritools.urisplit(uri)
    return parts.isuri()


def is_github_io_url(filename: str) -> bool:
    """
    Check if the given filename is a valid GitHub Pages URL.

    Args:
        filename (str): The filename to check.

    Returns:
        bool: True if the filename is a valid GitHub Pages URL, False otherwise.
    """
    if re.match(WEBBASE_LOADER_PATTERN, filename) and is_valid_uri(filename):
        LOGGER.debug("selected filetype github.io url, using WebBaseLoader(filename)")
        return True
    return False


def get_rag_loader(filename: str) -> TextLoader | PyMuPDFLoader | WebBaseLoader | None:
    """Get the appropriate loader for the given filename.

    Args:
        filename: The name of the file.

    Returns:
        The loader for the given file type, or None if the file type is not supported.
    """
    if is_github_io_url(f"{filename}"):
        return WebBaseLoader(
            web_paths=(f"{filename}",),
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
        )
    elif is_txt(filename):
        LOGGER.debug("selected filetype txt, using TextLoader(filename)")
        return TextLoader(filename)
    elif is_pdf(filename):
        LOGGER.debug("selected filetype pdf, using PyMuPDFLoader(filename)")
        return PyMuPDFLoader(filename, extract_images=True)
    else:
        LOGGER.debug(f"selected filetype UNKNOWN, using None. uri: {filename}")
        return None


def get_rag_splitter(filename: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> CharacterTextSplitter | None:
    """
    Get the appropriate text splitter for the given filename.

    This function determines the type of the given filename and returns the
    appropriate text splitter for it. It supports splitting text files and
    URLs matching the pattern for GitHub Pages.

    Args:
        filename (str): The name of the file to split.

    Returns:
        CharacterTextSplitter | None: The text splitter for the given file,
        or None if the file type is not supported.
    """
    LOGGER.debug(f"get_rag_splitter(filename={filename}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")

    if is_github_io_url(f"{filename}"):
        LOGGER.debug(
            f"selected filetype github.io url, usingRecursiveCharacterTextSplitter(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
        )
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif is_txt(filename):
        LOGGER.debug(f"selected filetype txt, using CharacterTextSplitter(chunk_size={chunk_size}, chunk_overlap=0)")
        return CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    else:
        LOGGER.debug(f"selected filetype UNKNOWN, using None. uri: {filename}")
        return None


def get_rag_embedding_function(
    filename: str, disallowed_special: Union[Literal["all"], set[str], Sequence[str], None] = None
) -> SentenceTransformerEmbeddings | OpenAIEmbeddings | None:
    """
    Get the appropriate embedding function for the given filename.

    This function determines the type of the given filename and returns the
    appropriate embedding function for it. It supports embedding text files,
    PDF files, and URLs matching the pattern for GitHub Pages.

    Args:
        filename (str): The name of the file to embed.

    Returns:
        SentenceTransformerEmbeddings | OpenAIEmbeddings | None: The embedding function for the given file,
        or None if the file type is not supported.
    """

    if is_github_io_url(f"{filename}"):
        LOGGER.debug(
            f"selected filetype github.io url, using OpenAIEmbeddings(disallowed_special={disallowed_special})"
        )
        return OpenAIEmbeddings(disallowed_special=disallowed_special)
    elif is_txt(filename):
        LOGGER.debug(
            f'selected filetype txt, using SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", disallowed_special={disallowed_special})'
        )
        return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    elif is_pdf(filename):
        LOGGER.debug(f"selected filetype pdf, using OpenAIEmbeddings(disallowed_special={disallowed_special})")
        return OpenAIEmbeddings(disallowed_special=disallowed_special)
    else:
        LOGGER.debug(f"selected filetype UNKNOWN, using None. uri: {filename}")
        return None


def compute_hash(text: str) -> str:
    """Compute hash of the text."""
    return hashlib.sha256(text.encode()).hexdigest()


def get_chunks_to_delete(
    chunks_prev: list[Document], chunks_current: list[Document], expired_days: float
) -> tuple[list[Document], list[Document]]:
    """
    Identifies chunks to be deleted based on their last seen timestamp and presence in the current run.

    Compare the chunks from the previous and current runs and identify chunks that are not present
    in the current run and have not been updated within the specified 'expired_days'. These chunks are marked for deletion.
    """
    ids_current = {d.metadata["item_id"] for d in chunks_current}

    ts_expired = int(datetime.now(timezone.utc).timestamp() - expired_days * DAY_IN_SECONDS)
    chunks_expired_delete, chunks_old_keep = [], []

    # chunks that have been crawled in the current run and are older than ts_expired => to delete
    for d in chunks_prev:
        if d.metadata["item_id"] not in ids_current:
            if d.metadata["last_seen_at"] < ts_expired:
                chunks_expired_delete.append(d)
            else:
                chunks_old_keep.append(d)

    return chunks_expired_delete, chunks_old_keep


def get_chunks_to_update(
    chunks_prev: list[Document], chunks_current: list[Document]
) -> tuple[list[Document], list[Document]]:
    """
    Identifies chunks that need to be updated or added based on their unique identifiers and checksums.

    Compare the chunks from the previous and current runs and identify chunks that are new or have
    undergone content changes by comparing their checksums. These chunks are marked for addition. chunks that are
    present in both runs but have not undergone content changes are marked for metadata update.
    """

    prev_id_checksum = defaultdict(list)
    for chunk in chunks_prev:
        prev_id_checksum[chunk.metadata["item_id"]].append(chunk.metadata["checksum"])

    chunks_add = []
    chunks_update_metadata = []
    for chunk in chunks_current:
        if chunk.metadata["item_id"] in prev_id_checksum:
            if chunk.metadata["checksum"] in prev_id_checksum[chunk.metadata["item_id"]]:
                chunks_update_metadata.append(chunk)
            else:
                chunks_add.append(chunk)
        else:
            chunks_add.append(chunk)

    return chunks_add, chunks_update_metadata


def add_item_last_seen_at(items: list[Document]) -> list[Document]:
    """Add last_seen_at timestamp to the metadata of each dataset item."""
    for item in items:
        item.metadata["last_seen_at"] = int(datetime.now(timezone.utc).timestamp())
    return items


def add_item_checksum(items: list[Document], dataset_fields_to_item_id: list[str]) -> list[Document]:
    """
    Adds a checksum and unique item_id to the metadata of each dataset item.

    This function computes a checksum for each item based on its content and metadata, excluding certain keys.
    The checksum is then added to the document's metadata. Additionally, a unique item ID is generated based on
    specified keys in the document's metadata and added to the metadata as well.
    """
    for item in items:
        item.metadata["checksum"] = compute_hash(item.json(exclude=EXCLUDE_KEYS_FROM_CHECKSUM))  # type: ignore[arg-type]
        item.metadata["item_id"] = compute_hash("".join([item.metadata[key] for key in dataset_fields_to_item_id]))

    return add_item_last_seen_at(items)


def add_chunk_id(chunks: list[Document]) -> list[Document]:
    """For every chunk (document stored in vector db) add chunk_id to metadata.

    The chunk_id is a unique identifier for each chunk and is not required but it is better to keep it in metadata.
    """
    for d in chunks:
        d.metadata["chunk_id"] = d.metadata.get("chunk_id", str(uuid4()))
    return chunks


# SOURCE: https://github.com/divyeg/meakuchatbot_project/blob/0c4483ce4bebce923233cf2a1139f089ac5d9e53/createVectorDB.ipynb#L203
def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """
    Calculate chunk IDs for a list of document chunks.

    This function calculates chunk IDs in the format "data/monopoly.pdf:6:2",
    where "data/monopoly.pdf" is the page source, "6" is the page number, and
    "2" is the chunk index.

    Args:
        chunks (list[Document]): The list of document chunks.

    Returns:
        list[Document]: The list of document chunks with chunk IDs added to their metadata.
    """
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    # USAGE: chunks_with_ids = calculate_chunk_ids(chunks)

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        # LOGGER.debug(f"chunk_id: {chunk_id}")
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks
