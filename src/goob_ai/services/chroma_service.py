"""goob_ai.services.chroma_service"""

# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportCallInDefaultInitializer=false
# pylint: disable=no-name-in-module

# pylint: disable=no-member
# LINK: https://github.com/mlsmall/RAG-Application-with-LangChain
# SOURCE: https://www.linkedin.com/pulse/building-retrieval-augmented-generation-rag-app-langchain-tiwari-stpfc/
# NOTE: This might be the one, take inspiration from the others
from __future__ import annotations

import argparse
import asyncio
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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Set, Union

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
from goob_ai.gen_ai.utilities import (
    WEBBASE_LOADER_PATTERN,
    calculate_chunk_ids,
    franchise_metadata,
    generate_document_hashes,
    get_file_extension,
    get_nested_value,
    get_rag_embedding_function,
    get_rag_loader,
    get_rag_splitter,
    get_suffix,
    is_github_io_url,
    is_pdf,
    is_txt,
    is_valid_uri,
    markdown_to_documents,
    remove_leading_period,
    string_to_doc,
    stringify_dict,
)
from goob_ai.services import (
    answer_question_from_context,
    bm25_retrieval,
    create_question_answer_from_context_chain,
    encode_from_string,
    encode_pdf,
    read_pdf_to_string,
    replace_t_with_space,
    retrieve_context_per_question,
    show_context,
    text_wrap,
)
from goob_ai.utils import file_functions

# from langchain_community.vectorstores import Chroma
# from langchain.vectorstores.chroma import Chroma

HERE = os.path.dirname(__file__)

DATA_PATH = os.path.join(HERE, "..", "data", "chroma", "documents")
CHROMA_PATH = os.path.join(HERE, "..", "data", "chroma", "vectorstorage")
CHROMA_PATH_API = Path(CHROMA_PATH)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Define the regex pattern to match a valid URL containing "github.io"
WEBBASE_LOADER_PATTERN = r"^https?://[a-zA-Z0-9.-]+\.github\.io(/.*)?$"


# TODO: Reimplement this function without langchain_chroma dependency
# async def llm_query(
#     collection_name: str = "",
#     question: str = "",
#     threshold: float = 0.65,
#     count: int = 5,
#     disallowed_special: Literal["all"] | set[str] | Sequence[str] = (),
# ) -> tuple[str | list | None, list[str | None]]:
#     LOGGER.debug(f"Querying chroma db. Count={count} Threshold={threshold}", collection_name=collection_name)
#
#     http_client = httpx.AsyncClient()
#
#     # Load DB
#     llm_db = Chroma(
#         collection_name=collection_name,
#         persist_directory=str(CHROMA_PATH),
#         embedding_function=OpenAIEmbeddings(
#             openai_api_key=aiosettings.openai_api_key.get_secret_value(),
#             disallowed_special=disallowed_special,
#             async_client=httpx.AsyncClient(),
#         ),
#     )
#
#     if not llm_db:
#         raise RuntimeError("Chroma DB does not exist")
#
#     # Search the DB.
#     similar = llm_db.similarity_search_with_relevance_scores(question, k=count)
#
#     if not similar:
#         LOGGER.debug("Unable to find matching results.", collection_name=collection_name)
#         return (None, [])
#
#     LOGGER.debug(f"Similar result count: {len(similar)}", collection_name=collection_name)
#
#     results: list[tuple[Document, float]] = []
#     for r in similar:
#         LOGGER.debug(f"Result Threshold: {r[1]:.4f}", collection_name=collection_name)
#         if r[1] > threshold and r not in results:
#             results.append(r)
#
#     if not results:
#         LOGGER.debug("Unable to find matching results.", collection_name=collection_name)
#         return (None, [])
#
#     LOGGER.debug(f"Final Result count: {len(results)}", collection_name=collection_name)
#
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=question)
#     LOGGER.debug(prompt)
#
#     # FIXME: This is a hack to get the model to work. We need to fix this in the llm_manager.
#     # model: ChatOpenAI | None = llm_manager.LlmManager().llm
#     model = ChatOpenAI(
#         name="ChatOpenAI",
#         model=aiosettings.chat_model,
#         streaming=True,
#         temperature=aiosettings.llm_temperature,
#         async_client=http_client,
#     )
#     response_text = model.invoke(prompt)
#
#     await http_client.aclose()
#
#     sources: list[str | None] = [doc.metadata.get("source") for doc, _score in results]
#     if not response_text.content:
#         return (None, [])
#     LOGGER.debug(f"LLM Response: {response_text.content}", collection_name=collection_name)
#     if len(response_text.content) > 2000:
#         return ("Sorry that response is too long for me to put in discord.", [])
#
#     await LOGGER.complete()
#     return (response_text.content, sources)


# # SOURCE: https://github.com/RSC-NA/rsc/blob/69f8ce29a6e38a960515564bf84fbd1d809468d8/rsc/llm/create_db.py#L176
# async def string_to_doc(text: str) -> Document:
#     """
#     Convert a string to a Document object.

#     Args:
#         text (str): The input string to convert.

#     Returns:
#         Document: The converted Document object.
#     """
#     return Document(page_content=text)


# # SOURCE: https://github.com/RSC-NA/rsc/blob/69f8ce29a6e38a960515564bf84fbd1d809468d8/rsc/llm/create_db.py#L176
# async def markdown_to_documents(docs: list[Document]) -> list[Document]:
#     """
#     Split Markdown documents into smaller chunks.

#     Args:
#         docs (list[Document]): The list of Markdown documents to split.

#     Returns:
#         list[Document]: The list of split document chunks.
#     """
#     md_splitter = MarkdownTextSplitter()
#     return md_splitter.split_documents(docs)


# # SOURCE: https://github.com/RSC-NA/rsc/blob/69f8ce29a6e38a960515564bf84fbd1d809468d8/rsc/llm/create_db.py#L176
# def franchise_metadata(record: dict, metadata: dict) -> dict:
#     """
#     Update the metadata dictionary with franchise-specific information.

#     Args:
#         record (dict): The record dictionary.
#         metadata (dict): The metadata dictionary to update.

#     Returns:
#         dict: The updated metadata dictionary.
#     """
#     LOGGER.debug(f"Metadata: {metadata}")
#     metadata["source"] = "API"
#     return metadata


# # # SOURCE: https://github.com/RSC-NA/rsc/blob/69f8ce29a6e38a960515564bf84fbd1d809468d8/rsc/llm/create_db.py#L176
# # async def json_to_docs(data: str, jq_schema: str, metadata_func: Callable | None) -> list[Document]:
# #     """
# #     Convert JSON data to a list of Document objects.

# #     Args:
# #         data (str): The JSON data as a string.
# #         jq_schema (str): The jq schema to apply to the JSON data.
# #         metadata_func (Callable | None): The function to apply to the metadata.

# #     Returns:
# #         list[Document]: The list of converted Document objects.
# #     """
# #     with tempfile.NamedTemporaryFile() as fd:
# #         if isinstance(data, str):
# #             fd.write(data.encode("utf-8"))
# #         elif isinstance(data, bytes):
# #             fd.write(data)
# #         else:
# #             raise TypeError("JSON data must be str or bytes")
# #         loader = JSONLoader(
# #             file_path=fd.name,
# #             jq_schema=jq_schema,
# #             text_content=False,
# #             metadata_func=franchise_metadata,
# #         )
# #         chunks = loader.load()

# #     return chunks


# # SOURCE: https://github.com/RSC-NA/rsc/blob/69f8ce29a6e38a960515564bf84fbd1d809468d8/rsc/llm/create_db.py#L176
# async def generate_document_hashes(docs: list[Document]) -> list[str]:
#     """
#     Generate hashes for a list of Document objects.

#     Args:
#         docs (list[Document]): The list of Document objects to generate hashes for.

#     Returns:
#         list[str]: The list of generated hashes.
#     """
#     hashes = []
#     for doc in docs:
#         source = doc.metadata.get("source")
#         api_id = doc.metadata.get("id")

#         if source and api_id:
#             ident = f"{source}/{api_id}"
#         elif source:
#             ident = f"{source}"
#         elif api_id:
#             LOGGER.warning(f"LLM Document has no source: {doc.page_content[:50]}")
#             ident = f"{api_id}"
#         else:
#             LOGGER.warning(f"LLM Document has no metadata: {doc.page_content[:50]}")
#             ident = doc.page_content

#         hash = hashlib.sha256(ident.encode("utf-8")).hexdigest()
#         hashes.append(hash)

#     await LOGGER.complete()
#     return hashes


# SOURCE: https://github.com/RSC-NA/rsc/blob/69f8ce29a6e38a960515564bf84fbd1d809468d8/rsc/llm/create_db.py#L176
# TODO: Reimplement this function without langchain_chroma dependency
# async def create_chroma_db(
#     collection_name: str,
#     docs: list[Document],
#     disallowed_special: Literal["all"] | set[str] | Sequence[str] = (),
#     reset: bool = False,
# ) -> None:
#     """
#     Create a Chroma database with the given collection name and documents.
#
#     Args:
#         collection_name (str): The name of the collection to create.
#         docs (list[Document]): The list of documents to add to the collection.
#         disallowed_special (Literal["all"] | set[str] | Sequence[str], optional): Special tokens to disallow.
#             Defaults to ().
#         reset (bool, optional): Whether to reset the database before creating it. Defaults to False.
#     """
#     if reset:
#         await rm_chroma_db()
#
#     if not CHROMA_PATH_API.absolute().exists():
#         # Create the directory if it doesn't exist
#         LOGGER.debug("Creating Chroma DB Directory", collection_name=collection_name)
#         CHROMA_PATH_API.absolute().mkdir(parents=True, exist_ok=True)
#
#     # Save the documents to the database
#     LOGGER.debug("Saving Chroma DB.", collection_name=collection_name)
#     Chroma.from_documents(
#         documents=docs,
#         collection_name=collection_name,
#         embedding=OpenAIEmbeddings(
#             openai_api_key=aiosettings.openai_api_key.get_secret_value(),
#             disallowed_special=disallowed_special,
#             async_client=httpx.AsyncClient(),
#         ),
#         persist_directory=str(CHROMA_PATH_API.absolute()),
#     )
#     LOGGER.info(f"Saved {len(docs)} chunks to {CHROMA_PATH_API}.", collection_name=collection_name)
#     await LOGGER.complete()


# SOURCE: https://github.com/RSC-NA/rsc/blob/69f8ce29a6e38a960515564bf84fbd1d809468d8/rsc/llm/create_db.py#L176
async def rm_chroma_db() -> None:
    """
    Remove the Chroma database directory if it exists.

    This function checks if the Chroma database directory exists and is a directory
    with the name "db". If the conditions are met, it deletes the directory and its
    contents using `shutil.rmtree()`. After deleting the directory, it waits for 5
    seconds using `asyncio.sleep()`.
    """
    if CHROMA_PATH_API.exists() and CHROMA_PATH_API.is_dir() and CHROMA_PATH_API.name == "vectorstorage":
        LOGGER.debug(f"Deleting Chroma DB directory: {CHROMA_PATH_API.absolute()}")
        shutil.rmtree(CHROMA_PATH_API.absolute())
        await asyncio.sleep(5)
    await LOGGER.complete()


# SOURCE: https://github.com/divyeg/meakuchatbot_project/blob/0c4483ce4bebce923233cf2a1139f089ac5d9e53/createVectorDB.ipynb#L203
def compare_two_words(w1: str, w2: str) -> None:
    """
    Compare the embeddings of two words.

    Args:
        w1 (str): The first word to compare.
        w2 (str): The second word to compare.
    """
    # Get embedding for a word.
    embedding_function = OpenAIEmbeddings()
    vector = embedding_function.embed_query(w1)
    LOGGER.info(f"Vector for '{w1}': {vector}")
    LOGGER.info(f"Vector length: {len(vector)}")

    # Compare vector of two words
    evaluator = load_evaluator("pairwise_embedding_distance")
    words = (w1, w2)
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])  # type: ignore
    LOGGER.info(f"Comparing ({words[0]}, {words[1]}): {x}")


# SOURCE: https://github.com/divyeg/meakuchatbot_project/blob/0c4483ce4bebce923233cf2a1139f089ac5d9e53/createVectorDB.ipynb#L203
# TODO: Enable and refactor this function
@pysnooper.snoop()
def add_or_update_documents(
    chunks: list[Document],
    persist_directory: str = CHROMA_PATH,
    disallowed_special: Literal["all"] | set[str] | Sequence[str] | None = (),
    use_custom_openai_embeddings: bool = False,
    collection_name: str = "",
    # path_to_document: str = "",
    embedding_function: Any | None = OpenAIEmbeddings(),
) -> None:
    """
    Add or update documents in a Chroma database.

    This function loads documents from the specified path, splits them into chunks if necessary,
    and adds or updates them in the Chroma database. It uses the appropriate loader and text splitter
    based on the file type of the document.

    Args:
        persist_directory (str): The directory where the Chroma database is persisted. Defaults to CHROMA_PATH.
        disallowed_special (Union[Literal["all"], set[str], Sequence[str], None]): Special characters to disallow in the embeddings. Defaults to an empty tuple.
        use_custom_openai_embeddings (bool): Whether to use custom OpenAI embeddings. Defaults to False.
        collection_name (str): The name of the collection in the Chroma database. Defaults to an empty string.
        path_to_document (str): The path to the document to be added or updated. Defaults to an empty string.
        embedding_function (Any | None): The embedding function to use. Defaults to OpenAIEmbeddings().

    Returns:
        None
    """
    # NOTE: orig code
    # from langchain_community.embeddings import HuggingFaceEmbeddings
    # embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Log the input parameters for debugging purposes
    # LOGGER.debug(f"path_to_document = {path_to_document}")
    LOGGER.debug(f"collection_name = {collection_name}")
    LOGGER.debug(f"embedding_function = {embedding_function}")

    db = get_chroma_db(persist_directory, embedding_function, collection_name=collection_name)

    # Get the Chroma client
    client = ChromaService.get_client()
    # FIXME: We need to make embedding_function optional
    # Add or retrieve the collection with the specified name
    collection: chromadb.Collection = ChromaService.add_collection(collection_name)

    # # Load the document using the appropriate loader based on the file type
    # loader: TextLoader | PyMuPDFLoader | WebBaseLoader | None = get_rag_loader(path_to_document)
    # # Load the documents using the selected loader
    # documents: list[Document] = loader.load()

    # # If the file type is txt, split the documents into chunks
    # text_splitter = get_rag_splitter(path_to_document)
    # if text_splitter:
    #     # Split the documents into chunks using the text splitter
    #     chunks: list[Document] = text_splitter.split_documents(documents)
    # else:
    #     # If no text splitter is available, use the original documents
    #     chunks: list[Document] = documents  # type: ignore

    #################################

    # embedder = OpenAIEmbeddings(openai_api_key=aiosettings.openai_api_key.get_secret_value())

    # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedder)

    try:
        last_request_time = 0
        RATE_LIMIT_INTERVAL = 10

        chunks_with_ids: list[Document] = calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = db.get(include=[])  # IDs are always included by default
        LOGGER.debug(f"existing_items: {existing_items}")
        existing_ids = set(existing_items["ids"])
        LOGGER.debug(f"existing_ids: {existing_ids}")
        LOGGER.info(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            LOGGER.info(f"Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            docs_added = db.add_documents(new_chunks, ids=new_chunk_ids)
            LOGGER.info(f"Saved {len(docs_added)} chunks to {CHROMA_PATH}.")
            # db.persist()
        else:
            LOGGER.info("No new documents to add")

    except Exception as ex:
        print(f"{ex}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"Error Class: {ex.__class__}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        print(output)
        print(f"exc_type: {exc_type}")
        print(f"exc_value: {exc_value}")
        traceback.print_tb(exc_traceback)
        if aiosettings.dev_mode:
            bpdb.pm()

    retriever: VectorStoreRetriever = db.as_retriever()

    return retriever


# Number of existing documents in DB: 0
# Adding new documents: 10
# Saved 10 chunks to input_data/chroma.


# def get_suffix(filename: str) -> str:
#     """Get the file extension from the given filename.

#     Args:
#         filename: The name of the file.

#     Returns:
#         The file extension in lowercase without the leading period.
#     """
#     ext = get_file_extension(filename)
#     ext_without_period = remove_leading_period(ext)
#     LOGGER.debug(f"ext: {ext}, ext_without_period: {ext_without_period}")
#     return ext


# def get_file_extension(filename: str) -> str:
#     """Get the file extension from the given filename.

#     Args:
#         filename: The name of the file.

#     Returns:
#         The file extension in lowercase.
#     """
#     return pathlib.Path(filename).suffix.lower()


# def remove_leading_period(ext: str) -> str:
#     """Remove the leading period from the file extension.

#     Args:
#         ext: The file extension.

#     Returns:
#         The file extension without the leading period.
#     """
#     return ext.replace(".", "")


# def is_pdf(filename: str) -> bool:
#     """Check if the given filename has a PDF extension.

#     Args:
#         filename: The name of the file.

#     Returns:
#         True if the file has a PDF extension, False otherwise.
#     """
#     suffix = get_suffix(filename)
#     res = suffix in file_functions.PDF_EXTENSIONS
#     LOGGER.debug(f"res: {res}")
#     return res


# def is_txt(filename: str) -> bool:
#     """Check if the given filename has a text extension.

#     Args:
#         filename: The name of the file.

#     Returns:
#         True if the file has a text extension, False otherwise.
#     """
#     suffix = get_suffix(filename)
#     res = suffix in file_functions.TXT_EXTENSIONS
#     LOGGER.debug(f"res: {res}")
#     return res


# def is_valid_uri(uri: str) -> bool:
#     """
#     Check if the given URI is valid.

#     Args:
#         uri (str): The URI to check.

#     Returns:
#         bool: True if the URI is valid, False otherwise.
#     """
#     parts = uritools.urisplit(uri)
#     return parts.isuri()


# def is_github_io_url(filename: str) -> bool:
#     """
#     Check if the given filename is a valid GitHub Pages URL.

#     Args:
#         filename (str): The filename to check.

#     Returns:
#         bool: True if the filename is a valid GitHub Pages URL, False otherwise.
#     """
#     if re.match(WEBBASE_LOADER_PATTERN, filename) and is_valid_uri(filename):
#         LOGGER.debug("selected filetype github.io url, using WebBaseLoader(filename)")
#         return True
#     return False


# def get_rag_loader(filename: str) -> TextLoader | PyMuPDFLoader | WebBaseLoader | None:
#     """Get the appropriate loader for the given filename.

#     Args:
#         filename: The name of the file.

#     Returns:
#         The loader for the given file type, or None if the file type is not supported.
#     """
#     if is_github_io_url(f"{filename}"):
#         return WebBaseLoader(
#             web_paths=(f"{filename}",),
#             bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
#         )
#     elif is_txt(filename):
#         LOGGER.debug("selected filetype txt, using TextLoader(filename)")
#         return TextLoader(filename)
#     elif is_pdf(filename):
#         LOGGER.debug("selected filetype pdf, using PyMuPDFLoader(filename)")
#         return PyMuPDFLoader(filename, extract_images=True)
#     else:
#         LOGGER.debug(f"selected filetype UNKNOWN, using None. uri: {filename}")
#         return None


# def get_rag_splitter(filename: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> CharacterTextSplitter | None:
#     """
#     Get the appropriate text splitter for the given filename.

#     This function determines the type of the given filename and returns the
#     appropriate text splitter for it. It supports splitting text files and
#     URLs matching the pattern for GitHub Pages.

#     Args:
#         filename (str): The name of the file to split.

#     Returns:
#         CharacterTextSplitter | None: The text splitter for the given file,
#         or None if the file type is not supported.
#     """
#     LOGGER.debug(f"get_rag_splitter(filename={filename}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")

#     if is_github_io_url(f"{filename}"):
#         LOGGER.debug(
#             f"selected filetype github.io url, usingRecursiveCharacterTextSplitter(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
#         )
#         return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     elif is_txt(filename):
#         LOGGER.debug(f"selected filetype txt, using CharacterTextSplitter(chunk_size={chunk_size}, chunk_overlap=0)")
#         return CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
#     else:
#         LOGGER.debug(f"selected filetype UNKNOWN, using None. uri: {filename}")
#         return None


# def get_rag_embedding_function(
#     filename: str, disallowed_special: Union[Literal["all"], set[str], Sequence[str], None] = None
# ) -> SentenceTransformerEmbeddings | OpenAIEmbeddings | None:
#     """
#     Get the appropriate embedding function for the given filename.

#     This function determines the type of the given filename and returns the
#     appropriate embedding function for it. It supports embedding text files,
#     PDF files, and URLs matching the pattern for GitHub Pages.

#     Args:
#         filename (str): The name of the file to embed.

#     Returns:
#         SentenceTransformerEmbeddings | OpenAIEmbeddings | None: The embedding function for the given file,
#         or None if the file type is not supported.
#     """

#     if is_github_io_url(f"{filename}"):
#         LOGGER.debug(
#             f"selected filetype github.io url, using OpenAIEmbeddings(disallowed_special={disallowed_special})"
#         )
#         return OpenAIEmbeddings(disallowed_special=disallowed_special)
#     elif is_txt(filename):
#         LOGGER.debug(
#             f'selected filetype txt, using SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", disallowed_special={disallowed_special})'
#         )
#         return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#     elif is_pdf(filename):
#         LOGGER.debug(f"selected filetype pdf, using OpenAIEmbeddings(disallowed_special={disallowed_special})")
#         return OpenAIEmbeddings(disallowed_special=disallowed_special)
#     else:
#         LOGGER.debug(f"selected filetype UNKNOWN, using None. uri: {filename}")
#         return None


def get_client(
    host: str = aiosettings.chroma_host, port: int = aiosettings.chroma_port, **kwargs: Any
) -> chromadb.ClientAPI:
    """Get the ChromaDB client.

    Returns:
        The ChromaDB client.
    """
    return chromadb.HttpClient(
        host=host,
        port=port,
        settings=ChromaSettings(allow_reset=True, is_persistent=True, persist_directory=CHROMA_PATH, **kwargs),
    )


# TODO: Reimplement this function without langchain_chroma dependency
# @pysnooper.snoop()
# def search_db(db: Chroma, query_text: str, k: int = 3) -> list[tuple[Document, float]] | None:
#     """Search the Chroma database for relevant documents.
#
#     Args:
#         db (Chroma): The Chroma database to search.
#         query_text (str): The query text to search for.
#         k (int, optional): The number of results to return. Defaults to 3.
#
#     Returns:
#         list[tuple[Document, float]] | None: A list of tuples containing the document and its relevance score,
#             or None if no results are found.
#     """
#     results = db.similarity_search_with_relevance_scores(query_text, k=k)
#     if not results:
#         return None
#     return results


# @pysnooper.snoop()
def generate_context_text(results: list[tuple[Document, float]]) -> str:
    """Generate the context text from the search results.

    Args:
        results (list[tuple[Document, float]]): The list of relevant documents and their scores.

    Returns:
        str: The generated context text.
    """
    return "\n\n---\n\n".join([doc.page_content for doc, _score in results])


def generate_prompt(context_text: str, query_text: str) -> str:
    """Generate the prompt for the model.

    Args:
        context_text (str): The context text generated from the search results.
        query_text (str): The query text to search for.

    Returns:
        str: The generated prompt.
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt_template.format(context=context_text, question=query_text)


@pysnooper.snoop()
def get_sources(results: list[tuple[Document, float]]) -> list[str | None]:
    """Get the sources from the search results.

    Args:
        results (list[tuple[Document, float]]): The list of relevant documents and their scores.

    Returns:
        list[str | None]: The list of sources.
    """
    return [doc.metadata.get("source", None) for doc, _score in results]


# TODO: Reimplement these functions without langchain_chroma dependency
# def get_response(
#     query_text: str,
#     persist_directory: str = CHROMA_PATH,
#     embedding_function: Any = OpenAIEmbeddings(),
#     model: Any = ChatOpenAI(),
#     k: int = 3,
#     collection_name: str = "",
#     reset: bool = False,
#     **kwargs: Any,
# ) -> str:
#     """Get a response from the LLM using the given query text.
#
#     Args:
#         query_text (str): The query text to use.
#         persist_directory (str): The directory to persist the Chroma database.
#         embedding_function (Any, optional): The embedding function to use. Defaults to OpenAIEmbeddings().
#         model (Any, optional): The model to use. Defaults to ChatOpenAI().
#         k (int, optional): The number of results to return. Defaults to 3.
#         collection_name (str, optional): The name of the collection. Defaults to "".
#         reset (bool, optional): Whether to reset the database. Defaults to False.
#         **kwargs: Additional keyword arguments to pass to Chroma.
#
#     Returns:
#         str: The response from the LLM.
#     """
#     db = get_chroma_db(persist_directory, embedding_function, collection_name=collection_name)
#     results = search_db(db, query_text, k=k)
#     if not results:
#         return "I don't know."
#     context_text = generate_context_text(results)
#     prompt = generate_prompt(context_text, query_text)
#     response = model.invoke(prompt)
#     return response.content

# def generate_and_query_data_store(
#     collection_name: str = "", embedding_function: Any = OpenAIEmbeddings(), reset: bool = False
# ) -> VectorStoreRetriever:
#     """Generate and store document embeddings in a Chroma vector store.
#
#     Args:
#         collection_name (str, optional): The name of the collection. Defaults to "".
#         embedding_function (Any, optional): The embedding function to use. Defaults to OpenAIEmbeddings().
#         reset (bool, optional): Whether to reset the database. Defaults to False.
#
#     Returns:
#         VectorStoreRetriever: The retriever for the database.
#     """
#     return generate_data_store(collection_name=collection_name, embedding_function=embedding_function, reset=reset)

# @pysnooper.snoop()
# def save_to_chroma(
#     chunks: list[Document],
#     disallowed_special: Literal["all"] | set[str] | Sequence[str] | None = (),
#     use_custom_openai_embeddings: bool = False,
#     collection_name: str = "",
#     reset: bool = False,
# ) -> VectorStoreRetriever:
#     """Save document chunks to a Chroma vector store.
#
#     Args:
#         chunks (list[Document]): The document chunks to save.
#         disallowed_special (Literal["all"] | set[str] | Sequence[str] | None, optional): Special tokens to disallow.
#             Defaults to ().
#         use_custom_openai_embeddings (bool, optional): Whether to use custom OpenAI embeddings. Defaults to False.
#         collection_name (str, optional): The name of the collection. Defaults to "".
#         reset (bool, optional): Whether to reset the database. Defaults to False.
#
#     Returns:
#         VectorStoreRetriever: The retriever for the database.
#     """
#     if reset:
#         clean_chroma_db(collection_name=collection_name)
#
#     if use_custom_openai_embeddings:
#         embeddings = CustomOpenAIEmbeddings(disallowed_special=disallowed_special)
#     else:
#         embeddings = OpenAIEmbeddings(disallowed_special=disallowed_special)
#
#     # Create directory if needed
#     if not CHROMA_PATH_API.absolute().exists():
#         # Create brand new DB if it doesn't exist
#         LOGGER.debug("Creating Chroma DB Directory", collection_name=collection_name)
#         CHROMA_PATH_API.absolute().mkdir(parents=True, exist_ok=True)
#
#     # Save the documents to the database
#     LOGGER.debug("Saving Chroma DB.", collection_name=collection_name)
#     db = Chroma.from_documents(
#         documents=chunks,
#         collection_name=collection_name,
#         embedding=embeddings,
#         persist_directory=str(CHROMA_PATH_API.absolute()),
#     )
#     LOGGER.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH_API}.", collection_name=collection_name)
#     return db.as_retriever()


def main() -> None:
    """
    Main function to generate and store document embeddings.

    This function initializes the process of generating and storing document embeddings
    in a Chroma vector store. It calls the `generate_data_store` function to perform
    the necessary steps.
    """
    generate_data_store()


class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    """Custom embeddings class using OpenAI's API.

    This class extends the OpenAIEmbeddings class to provide custom functionality
    for embedding documents using OpenAI's API.

    Attributes:
        openai_api_key (str): The API key for accessing OpenAI services.
    """

    def __init__(
        self,
        openai_api_key: str = aiosettings.openai_api_key.get_secret_value(),
        disallowed_special: Literal["all"] | set[str] | Sequence[str] | None = None,
    ) -> None:
        """Initialize the CustomOpenAIEmbeddings class.

        Args:
            openai_api_key (str): The API key for accessing OpenAI services.
        """
        super().__init__(openai_api_key=openai_api_key, disallowed_special=disallowed_special)

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        This method takes a list of document texts and returns their embeddings
        as a list of float vectors.

        Args:
            texts (list of str): The list of document texts to be embedded.

        Returns:
            list of list of float: The embeddings of the input documents.
        """
        return super().embed_documents(texts)

    def __call__(self, input: list[str]) -> list[float]:
        """Embed a list of documents.

        This method is a callable that takes a list of document texts and returns
        their embeddings as a list of float vectors.

        Args:
            input (list of str): The list of document texts to be embedded.

        Returns:
            list of float: The embeddings of the input documents.
        """
        return self._embed_documents(input)


@pysnooper.snoop()
def generate_data_store(
    collection_name: str = "", embedding_function: Any = OpenAIEmbeddings(), reset: bool = False
) -> VectorStoreRetriever:
    """Generate and store document embeddings in a Chroma vector store.

    This function performs the following steps:
    1. Loads documents from the specified data path.
    2. Splits the loaded documents into smaller chunks.
    3. Saves the chunks into a Chroma vector store for efficient retrieval.
    """
    if reset:
        _ = clean_chroma_db(collection_name=collection_name)

    documents = load_documents()
    chunks = split_text(documents)
    retriever: VectorStoreRetriever = save_to_chroma(chunks, collection_name=collection_name, reset=reset)
    return retriever


@pysnooper.snoop()
def do_generate_data_store_and_update(
    collection_name: str = "",
    embedding_function: Any = OpenAIEmbeddings(),
    reset: bool = False,
    disallowed_special: Literal["all"] | set[str] | Sequence[str] | None = (),
    use_custom_openai_embeddings: bool = False,
) -> VectorStoreRetriever:
    if reset:
        _ = clean_chroma_db(collection_name=collection_name)

    documents = load_documents()
    chunks = split_text(documents)
    # retriever: VectorStoreRetriever = save_to_chroma(chunks, collection_name=collection_name, reset=reset)
    retriever: VectorStoreRetriever = add_or_update_documents(chunks, collection_name=collection_name)
    return retriever


def clean_chroma_db(collection_name: str = "") -> None:
    LOGGER.info(f"Resetting ChromaDB at {CHROMA_PATH} ...")
    client = get_client()
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        # time.sleep(5)

        client = get_client()
        _await_server(api=client)
        LOGGER.info("Resetting ChromaDB")
        # client: chromadb.ClientAPI = get_client()
        # client.reset()
        # Create brand new DB if it doesn't exist
        LOGGER.debug("Creating Chroma DB Directory", collection_name=collection_name)
        CHROMA_PATH_API.absolute().mkdir(parents=True, exist_ok=True)
        LOGGER.debug("sleeping for 5 seconds")
        # time.sleep(5)
        # await asyncio.sleep(5)
        client = get_client()
        col = client.get_or_create_collection(name=collection_name)
        count = col.count()
        LOGGER.info(f"Collection {collection_name} has {count} documents.")
    return client


def generate_and_query_data_store(
    collection_name: str = "", embedding_function: Any = OpenAIEmbeddings(), reset: bool = False
) -> VectorStoreRetriever:
    retriever: VectorStoreRetriever = generate_data_store(
        collection_name=collection_name, embedding_function=embedding_function, reset=reset
    )
    return retriever


@pysnooper.snoop()
def load_documents(data_path: str = DATA_PATH) -> list[Document]:
    """Load documents from the specified data path.

    This function loads documents from the specified data path and returns them
    as a list of Document objects.

    Returns:
        List[Document]: The list of loaded documents.
    """
    documents = []

    d = file_functions.tree(data_path)
    result_pdfs = file_functions.filter_pdfs(d)
    # TODO: add txt files via something like this:
    # result_txts = file_functions.filter_txts(d)
    # results = result_pdfs + result_txts

    # LOGGER.info(f"Found {len(results)} documents")
    LOGGER.info(f"Found {len(result_pdfs)} documents")

    # for filename in results:
    for filename in result_pdfs:
        LOGGER.info(f"Loading document: {filename}")
        # loader = PyMuPDFLoader(f"{filename}", extract_images=True)
        loader: TextLoader | PyMuPDFLoader | WebBaseLoader | None = get_rag_loader(filename)
        LOGGER.info(f"Loader: {loader}")
        documents.extend(loader.load())
    return documents


# @pysnooper.snoop()
def split_text(
    documents: list[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 100,
    length_function: Callable[[Document], int] = len,
    add_start_index: bool = True,
) -> list[Document]:
    """Split documents into smaller chunks.

    This function takes a list of documents and splits each document into smaller chunks
    using the RecursiveCharacterTextSplitter. The chunks are then returned as a list of
    Document objects.

    Args:
        documents (List[Document]): The list of documents to be split into chunks.

    Returns:
        List[Document]: The list of document chunks.
    """
    LOGGER.debug(
        f"Split text with chunk size: {chunk_size}, chunk overlap: {chunk_overlap}, length function: {length_function}, add start index: {add_start_index}"
    )

    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=add_start_index,
    )
    # aka chunks = all_splits
    chunks: list[Document] = text_splitter.split_documents(documents)
    # cleaned_chunks = replace_t_with_space(chunks)
    LOGGER.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


@pysnooper.snoop()
def save_to_chroma(
    chunks: list[Document],
    disallowed_special: Literal["all"] | set[str] | Sequence[str] | None = (),
    use_custom_openai_embeddings: bool = False,
    collection_name: str = "",
    reset: bool = False,
) -> VectorStoreRetriever:
    """Save document chunks to a Chroma vector store.

    This function performs the following steps:
    1. Initializes the embeddings using the OpenAI API key.
    2. Creates a new Chroma database from the document chunks.
    3. Persists the database to the specified directory.

    Args:
        chunks (list of Document): The list of document chunks to be saved.
    """
    # Clear out the database first.

    # DISABLED: do this way before this step otherwise, race condition.
    # if reset:
    #     LOGGER.info(f"Resetting ChromaDB at {CHROMA_PATH} ...")
    #     if os.path.exists(CHROMA_PATH):
    #         shutil.rmtree(CHROMA_PATH)
    #         # time.sleep(5)
    #         client = get_client()
    #         _await_server(api=client)
    #         LOGGER.info("Resetting ChromaDB")
    #         # client: chromadb.ClientAPI = get_client()
    #         # client.reset()
    #         # Create brand new DB if it doesn't exist
    #         LOGGER.debug("Creating Chroma DB Directory", collection_name=collection_name)
    #         CHROMA_PATH_API.absolute().mkdir(parents=True, exist_ok=True)
    #         LOGGER.debug("sleeping for 5 seconds")
    #         # time.sleep(5)
    #         # await asyncio.sleep(5)
    #         client = get_client()
    #         col = client.get_or_create_collection(name=collection_name)
    #         count = col.count()
    #         LOGGER.info(f"Collection {collection_name} has {count} documents.")

    # Create directory if needed
    # if not CHROMA_PATH_API.absolute().exists():

    # default embeddings
    LOGGER.error("default embeddings OpenAIEmbeddings")
    embeddings = OpenAIEmbeddings(
        openai_api_key=aiosettings.openai_api_key.get_secret_value(), disallowed_special=disallowed_special
    )

    # if flag set to use custom embeddings, override
    if use_custom_openai_embeddings:
        LOGGER.error("Using CustomOpenAIEmbeddings")
        embeddings = CustomOpenAIEmbeddings(
            openai_api_key=aiosettings.openai_api_key.get_secret_value(), disallowed_special=disallowed_special
        )

    LOGGER.info(embeddings)

    # import bpdb

    # bpdb.set_trace()
    try:
        # Add to vectorDB
        # from_documents = Create a Chroma vectorstore from a list of documents.
        vectorstore: ChromaVectorStore = ChromaVectorStore.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH, collection_name=collection_name
        )
        db = vectorstore

        # vectorstore.persist()

        LOGGER.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

        # import bpdb

        # bpdb.set_trace()

        retriever: VectorStoreRetriever = vectorstore.as_retriever()

    except Exception as ex:
        print(f"{ex}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"Error Class: {ex.__class__}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        print(output)
        print(f"exc_type: {exc_type}")
        print(f"exc_value: {exc_value}")
        traceback.print_tb(exc_traceback)
        if aiosettings.dev_mode:
            bpdb.pm()

    return retriever


class ChromaService:
    """
    Service class for interacting with ChromaDB.

    This class provides static methods to interact with ChromaDB, including
    adding collections, listing collections, and retrieving collections.
    """

    client: chromadb.ClientAPI | None = get_client()
    collection: chromadb.Collection | None = None

    def __init__(self):
        # self.name = "ChromaService"
        # self.client = get_client()
        pass

    @staticmethod
    def add_collection(collection_name: str, embedding_function: Any | None = None) -> chromadb.Collection:
        """Add a collection to ChromaDB.

        Args:
            collection_name (str): The name of the collection to add.
            embedding_function (Any | None, optional): The embedding function to use. Defaults to None.

        Returns:
            chromadb.Collection: The created collection.
        """
        client = get_client()
        collection = client.get_or_create_collection(name=collection_name)
        return collection

    @staticmethod
    def get_list_collections() -> Sequence[chromadb.Collection]:
        """Get a list of all collections in ChromaDB.

        Returns:
            Sequence[chromadb.Collection]: The list of collections.
        """
        client = get_client()
        return client.list_collections()

    @staticmethod
    def get_collection(collection_name: str, embedding_function: Any) -> chromadb.Collection | None:
        """Get a collection from ChromaDB.

        Args:
            collection_name (str): The name of the collection to get.
            embedding_function (Any): The embedding function to use.

        Returns:
            chromadb.Collection | None: The collection if it exists, None otherwise.
        """
        client = get_client()
        try:
            collection = client.get_collection(name=collection_name)
            return collection
        except ValueError:
            return None

    @staticmethod
    def get_client() -> chromadb.ClientAPI:
        """Get the ChromaDB client.

        Returns:
            chromadb.ClientAPI: The ChromaDB client.
        """
        return get_client()

    @staticmethod
    def get_or_create_collection(collection_name: str, embedding_function: Any) -> chromadb.Collection:
        """Get or create a collection in ChromaDB.

        Args:
            collection_name (str): The name of the collection.
            embedding_function (Any): The embedding function to use.

        Returns:
            chromadb.Collection: The collection.
        """
        client = get_client()
        collection = client.get_or_create_collection(name=collection_name)
        return collection

    # TODO: Reimplement these methods without langchain_chroma dependency
    # @staticmethod
    # def get_response(query_text: str, collection_name: str = "", reset: bool = False) -> str:
    #     """Get a response from the LLM using the given query text.
    #
    #     Args:
    #         query_text (str): The query text to use.
    #         collection_name (str, optional): The name of the collection to use. Defaults to "".
    #         reset (bool, optional): Whether to reset the database. Defaults to False.
    #
    #     Returns:
    #         str: The response from the LLM.
    #     """
    #     return get_response(query_text, collection_name=collection_name, reset=reset)

    # @staticmethod
    # def generate_data_store() -> None:
    #     """Generate a data store from the documents."""
    #     generate_data_store()

    # @staticmethod
    # def load_documents() -> list[Document]:
    #     """Load documents from the data directory.
    #
    #     Returns:
    #         list[Document]: The loaded documents.
    #     """
    #     return load_documents()

    # @staticmethod
    # def add_and_query(collection_name: str = "", question: str = "", reset: bool = False) -> VectorStoreRetriever:
    #     """Add documents to the database and query it.
    #
    #     Args:
    #         collection_name (str, optional): The name of the collection. Defaults to "".
    #         question (str, optional): The question to query. Defaults to "".
    #         reset (bool, optional): Whether to reset the database. Defaults to False.
    #
    #     Returns:
    #         VectorStoreRetriever: The retriever for the database.
    #     """
    #     return generate_and_query_data_store(collection_name=collection_name, reset=reset)

    # @staticmethod
    # def split_text(documents: list[Document]) -> list[Document]:
    #     """Split text into chunks.
    #
    #     Args:
    #         documents (list[Document]): The documents to split.
    #
    #     Returns:
    #         list[Document]: The split documents.
    #     """
    #     return split_text(documents)

    # @staticmethod
    # def get_vector_store_from_client(
    #     collection_name: str | None = "",
    #     embedding_function: Any | None = None,
    #     client: chromadb.ClientAPI | None = None,
    # ) -> ChromaVectorStore:
    #     """Get a vector store from a ChromaDB client.
    #
    #     Args:
    #         collection_name (str | None, optional): The name of the collection. Defaults to "".
    #         embedding_function (Any | None, optional): The embedding function to use. Defaults to None.
    #         client (chromadb.ClientAPI | None, optional): The ChromaDB client. Defaults to None.
    #
    #     Returns:
    #         ChromaVectorStore: The vector store.
    #     """
    #     if client is None:
    #         client = get_client()
    #     collection = client.get_or_create_collection(name=collection_name)
    #     return ChromaVectorStore(collection=collection, embedding_function=embedding_function)

    # @staticmethod
    # def add_to_chroma(
    #     path_to_document: str = "", collection_name: str = "", embedding_function: Any | None = None
    # ) -> ChromaVectorStore:
    #     """Add a document to ChromaDB.
    #
    #     Args:
    #         path_to_document (str, optional): The path to the document. Defaults to "".
    #         collection_name (str, optional): The name of the collection. Defaults to "".
    #         embedding_function (Any | None, optional): The embedding function to use. Defaults to None.
    #
    #     Returns:
    #         ChromaVectorStore: The vector store containing the document.
    #     """
    #     # sourcery skip: inline-immediately-returned-variable
    #     client = get_client()
    #     collection = client.get_or_create_collection(name=collection_name)
    #     vector_store = ChromaVectorStore(collection=collection, embedding_function=embedding_function)
    #     return vector_store

    # @staticmethod
    # def save_to_chroma(chunks: list[Document]) -> None:
    #     """Save document chunks to ChromaDB.
    #
    #     Args:
    #         chunks (list[Document]): The document chunks to save.
    #     """
    #     save_to_chroma(chunks)


def _await_server(api: ServerAPI | ClientAPI = get_client(), attempts: int = 0) -> None:
    try:
        api.heartbeat()
    except ConnectError as e:
        LOGGER.warning(
            f"Error connecting to ChromaDB ... this is expected... retrying ... attempt {attempts} with client {api}: {e}"
        )
        if attempts > 15:
            raise e
        else:
            time.sleep(4)
            _await_server(api, attempts + 1)


if __name__ == "__main__":
    main()
