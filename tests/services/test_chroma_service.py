from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import shutil
import tempfile

from collections.abc import Generator, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Literal, Set, Union

from chromadb import Collection
from goob_ai.aio_settings import aiosettings
from goob_ai.services.chroma_service import (
    CHROMA_PATH,
    CHROMA_PATH_API,
    ChromaService,
    CustomOpenAIEmbeddings,
    add_or_update_documents,
    calculate_chunk_ids,
    compare_two_words,
    create_chroma_db,
    franchise_metadata,
    generate_and_query_data_store,
    generate_data_store,
    generate_document_hashes,
    get_chroma_db,
    get_client,
    get_file_extension,
    get_rag_embedding_function,
    get_rag_loader,
    get_rag_splitter,
    get_response,
    get_suffix,
    is_github_io_url,
    is_pdf,
    is_txt,
    is_valid_uri,
    load_documents,
    markdown_to_documents,
    rm_chroma_db,
    save_to_chroma,
    search_db,
    split_text,
    string_to_doc,
)
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_chroma import Chroma as ChromaVectorStore
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_text_splitters import MarkdownTextSplitter
from loguru import logger as LOGGER

import pytest


if TYPE_CHECKING:
    from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock

    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture

# import pysnooper
# import manhole
# # this will start the daemon thread
# manhole.install()


# def test_compare_two_words(caplog: LogCaptureFixture):
#     """
#     Test the compare_two_words function.
#     """
#     caplog.set_level(logging.DEBUG)
#     debug = [i.message for i in caplog.records if i.levelno == logging.DEBUG]
#     # Call the function with sample words
#     compare_two_words("apple", "banana")

#     # Check if the expected logs are present
#     assert "Vector for 'apple':" in caplog.text
#     assert "Vector length:" in caplog.text
#     assert "Comparing (apple, banana):" in caplog.text
#     caplog.clear()

FIRST_NAMES = ["Ada", "Bela", "Cade", "Dax", "Eva", "Fynn", "Gia", "Hugo", "Ivy", "Jax"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor"]


def generate_random_name():
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    return f"{first_name}_{last_name}".lower()


@pytest.mark.services()
def test_calculate_chunk_ids():
    """
    Test the calculate_chunk_ids function.
    """
    # Create sample document chunks
    chunks = [
        Document(page_content="Chunk 1", metadata={"source": "data/file1.pdf", "page": 1}),
        Document(page_content="Chunk 2", metadata={"source": "data/file1.pdf", "page": 1}),
        Document(page_content="Chunk 3", metadata={"source": "data/file1.pdf", "page": 2}),
        Document(page_content="Chunk 4", metadata={"source": "data/file2.pdf", "page": 1}),
    ]

    # Call the function
    result = calculate_chunk_ids(chunks)

    # Check if the chunk IDs are correctly assigned
    assert result[0].metadata["id"] == "data/file1.pdf:1:0"
    assert result[1].metadata["id"] == "data/file1.pdf:1:1"
    assert result[2].metadata["id"] == "data/file1.pdf:2:0"
    assert result[3].metadata["id"] == "data/file2.pdf:1:0"


@pytest.fixture()
def mock_openai_api_key(mocker: MockerFixture) -> str:
    """
    Fixture to provide a mock OpenAI API key for testing purposes.

    This fixture returns a mock OpenAI API key that can be used in tests
    to simulate the presence of a valid API key without making actual
    API calls.

    Args:
    ----
        mocker (MockerFixture): The mocker fixture for patching.

    Returns:
    -------
        str: A mock OpenAI API key.

    """
    return "test_api_key"


@pytest.fixture()
def mock_github_io_url(mocker: MockerFixture) -> str:
    """
    Fixture to github.io url for testing purposes.

    Args:
    ----
        mocker (MockerFixture): The mocker fixture for patching.

    Returns:
    -------
        str: github io url.

    """
    return "https://lilianweng.github.io/posts/2023-06-23-agent/"


@pytest.fixture()
def custom_embeddings(mock_openai_api_key: str) -> CustomOpenAIEmbeddings:
    """
    Create a CustomOpenAIEmbeddings instance with the provided API key.

    Args:
    ----
        mock_openai_api_key (str): The OpenAI API key to use for the embeddings.

    Returns:
    -------
        CustomOpenAIEmbeddings: An instance of CustomOpenAIEmbeddings initialized with the provided API key.

    """
    return CustomOpenAIEmbeddings(openai_api_key=mock_openai_api_key)


@pytest.mark.services()
@pytest.mark.parametrize(
    "filename, expected_extension",
    [
        ("example.txt", ".txt"),
        ("document.pdf", ".pdf"),
        ("image.jpg", ".jpg"),
        ("file.docx", ".docx"),
        ("archive.tar.gz", ".gz"),
        ("no_extension", ""),
    ],
)
def test_get_file_extension(filename: str, expected_extension: str) -> None:
    """
    Test the get_file_extension function.

    This test verifies that the `get_file_extension` function correctly extracts
    the file extension from the given filename.

    Args:
        filename (str): The filename to test.
        expected_extension (str): The expected file extension.
    """
    extension = get_file_extension(filename)
    assert extension == expected_extension


@pytest.mark.services()
def test_add_collection(mocker: MockerFixture) -> None:
    """
    Test the add_collection function of ChromaService.

    This test verifies that the add_collection function correctly adds a collection
    to ChromaDB using the provided collection name and embedding function.

    Args:
    ----
        mocker (MockerFixture): The mocker fixture for patching.

    """
    from goob_ai.services.chroma_service import ChromaService

    mock_client = mocker.patch.object(ChromaService, "client")
    mock_collection = mocker.Mock()
    mock_client.get_or_create_collection.return_value = mock_collection

    collection_name = "test_collection"
    embedding_function = mocker.Mock()

    result = ChromaService.add_collection(collection_name, embedding_function=embedding_function)

    assert result == mock_collection
    mock_client.get_or_create_collection.assert_called_once_with(
        name=collection_name, embedding_function=embedding_function
    )


@pytest.mark.services()
def test_get_client(mocker: MockerFixture) -> None:
    """
    Test the get_client function of ChromaService.

    This test verifies that the get_client function correctly retrieves
    the ChromaDB client.

    Args:
    ----
        mocker (MockerFixture): The mocker fixture for patching.

    """
    from goob_ai.services.chroma_service import ChromaService

    mock_client = mocker.Mock()
    mocker.patch.object(ChromaService, "client", mock_client)

    result = ChromaService.get_client()

    assert result == mock_client


@pytest.mark.services()
def test_get_collection(mocker: MockerFixture) -> None:
    """
    Test the get_collection function of ChromaService.

    This test verifies that the get_collection function correctly retrieves
    a collection from ChromaDB using the provided collection name and embedding function.

    Args:
    ----
        mocker (MockerFixture): The mocker fixture for patching.

    """
    from goob_ai.services.chroma_service import ChromaService

    mock_client = mocker.patch.object(ChromaService, "client")
    mock_collection = mocker.Mock()
    mock_client.get_collection.return_value = mock_collection

    collection_name = "test_collection"
    embedding_function = mocker.Mock()

    result = ChromaService.get_collection(collection_name, embedding_function)

    assert result == mock_collection
    mock_client.get_collection.assert_called_once_with(name=collection_name, embedding_function=embedding_function)


@pytest.mark.services()
def test_get_list_collections(mocker: MockerFixture) -> None:
    """
    Test the get_list_collections function of ChromaService.

    This test verifies that the get_list_collections function correctly retrieves
    the list of collections from ChromaDB.

    Args:
    ----
        mocker (MockerFixture): The mocker fixture for patching.

    """
    from goob_ai.services.chroma_service import ChromaService

    mock_client = mocker.patch.object(ChromaService, "client")
    mock_collections = [mocker.Mock(), mocker.Mock()]
    mock_client.list_collections.return_value = mock_collections

    result = ChromaService.get_list_collections()

    assert result == mock_collections
    mock_client.list_collections.assert_called_once()


@pytest.mark.services()
@pytest.mark.slow()
@pytest.mark.skipif(
    not os.getenv("DEBUG_AIDER"),
    reason="These tests are meant to only run locally on laptop prior to porting it over to new system",
)
def test_custom_openai_embeddings_call(mocker: MockerFixture, custom_embeddings: CustomOpenAIEmbeddings) -> None:
    """
    Test the call method of CustomOpenAIEmbeddings.

    This test verifies that the call method of CustomOpenAIEmbeddings returns
    the expected embeddings for the given texts.

    Args:
    ----
        mocker (MockerFixture): The mocker fixture for patching.
        custom_embeddings (CustomOpenAIEmbeddings): An instance of CustomOpenAIEmbeddings.

    """
    mock_texts: list[str] = ["This is a test document."]
    mock_embeddings: list[list[float]] = [[0.1, 0.2, 0.3]]

    mocker.patch.object(CustomOpenAIEmbeddings, "_embed_documents", return_value=mock_embeddings)

    result: list[list[float]] = custom_embeddings(mock_texts)
    assert result == mock_embeddings

    mock_texts = ["This is a test document."]
    mock_embeddings = [[0.1, 0.2, 0.3]]

    mocker.patch.object(CustomOpenAIEmbeddings, "_embed_documents", return_value=mock_embeddings)

    result = custom_embeddings(mock_texts)
    assert result == mock_embeddings


@pytest.fixture()
def mock_pdf_file(tmp_path: Path) -> Path:
    """
    Fixture to create a mock PDF file for testing purposes.

    This fixture creates a temporary directory and copies a test PDF file into it.
    The path to the mock PDF file is then returned for use in tests.

    Args:
    ----
        tmp_path (Path): The temporary path provided by pytest.

    Returns:
    -------
        Path: A Path object of the path to the mock PDF file.

    """
    test_pdf_path: Path = tmp_path / "rich-readthedocs-io-en-latest.pdf"
    shutil.copy("src/goob_ai/data/chroma/documents/rich-readthedocs-io-en-latest.pdf", test_pdf_path)
    return test_pdf_path


@pytest.fixture()
def mock_txt_file(tmp_path: Path) -> Path:
    """
    Fixture to create a mock text file for testing purposes.

    This fixture creates a temporary directory and copies a test txt file into it.
    The path to the mock txt file is then returned for use in tests.

    Args:
    ----
        tmp_path (Path): The temporary path provided by pytest.

    Returns:
    -------
        Path: A Path object of the path to the mock txt file.

    """
    test_txt_path: Path = tmp_path / "state_of_the_union.txt"
    shutil.copy("src/goob_ai/data/chroma/documents/state_of_the_union.txt", test_txt_path)
    return test_txt_path


@pytest.mark.services()
# @pytest.mark.vcr(allow_playback_repeats=True, match_on=["request_matcher"], ignore_localhost=False)
@pytest.mark.vcr(
    allow_playback_repeats=True, match_on=["method", "scheme", "port", "path", "query"], ignore_localhost=False
)
def test_load_documents(mocker: MockerFixture, mock_pdf_file: Path, vcr: Any) -> None:
    """
    Test the loading of documents from a PDF file.

    This test verifies that the `load_documents` function correctly loads
    documents from a PDF file, splits the text into chunks, and saves the
    chunks to Chroma.

    Args:
    ----
        mocker (MockerFixture): The mocker fixture for patching.
        mock_pdf_file (Path): The path to the mock PDF file.

    The test performs the following steps:
    1. Mocks the `os.listdir` and `os.path.join` functions to simulate the presence of the PDF file.
    2. Mocks the `PyPDFLoader` to return a document with test content.
    3. Calls the `generate_data_store` function to load, split, and save the document.
    4. Asserts that the document is loaded, split, and saved correctly.

    """

    from goob_ai.services.chroma_service import load_documents

    documents = load_documents()

    # this is a bad test, cause the data will change eventually. Need to find a way to test this.
    assert len(documents) == 680
    assert vcr.play_count == 0


@pytest.mark.services()
@pytest.mark.slow()
# @pytest.mark.skipif(
#     os.getenv("PINECONE_ENV"),
#     reason="These tests are meant to only run locally on laptop prior to porting it over to new system",
# )
def test_split_text(mocker: MockerFixture) -> None:
    """
    Test the split_text function.

    This test verifies that the `split_text` function correctly splits
    documents into chunks using the RecursiveCharacterTextSplitter.

    Args:
    ----
        mocker (MockerFixture): The mocker fixture for patching.

    The test performs the following steps:
    1. Mocks the RecursiveCharacterTextSplitter to return predefined chunks.
    2. Calls the `split_text` function with a mock document.
    3. Asserts that the document is split into the expected chunks.
    4. Verifies that the RecursiveCharacterTextSplitter is called with the correct arguments.

    """
    from typing import List

    from goob_ai.services.chroma_service import split_text
    from langchain.schema import Document

    mock_documents: list[Document] = [Document(page_content="This is a test document.", metadata={})]
    mock_chunks: list[Document] = [
        Document(page_content="This is a test", metadata={"start_index": 0}),
        Document(page_content="document.", metadata={"start_index": 15}),
    ]

    mock_text_splitter: MagicMock | AsyncMock | NonCallableMagicMock = mocker.patch(
        "goob_ai.services.chroma_service.RecursiveCharacterTextSplitter"
    )
    mock_text_splitter.return_value.split_documents.return_value = mock_chunks

    chunks: list[Document] = split_text(mock_documents)

    assert len(chunks) == 2
    assert chunks[0].page_content == "This is a test"
    assert chunks[1].page_content == "document."
    mock_text_splitter.return_value.split_documents.assert_called_once_with(mock_documents)
    mock_documents = [Document(page_content="This is a test document.", metadata={})]
    mock_chunks = [
        Document(page_content="This is a test", metadata={"start_index": 0}),
        Document(page_content="document.", metadata={"start_index": 15}),
    ]

    mock_text_splitter = mocker.patch("goob_ai.services.chroma_service.RecursiveCharacterTextSplitter")
    mock_text_splitter.return_value.split_documents.return_value = mock_chunks

    chunks = split_text(mock_documents)

    assert len(chunks) == 2
    assert chunks[0].page_content == "This is a test"
    assert chunks[1].page_content == "document."
    mock_text_splitter.return_value.split_documents.assert_called_once_with(mock_documents)


@pytest.mark.services()
# FIXME: This is a work in progress till I can incorporate this into the main codebase
@pytest.mark.slow()
@pytest.mark.integration()
@pytest.mark.e2e()
def test_chroma_service_e2e(mocker: MockerFixture, mock_txt_file: Path) -> None:
    import chromadb

    from goob_ai.services.chroma_service import ChromaService
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
    from langchain_text_splitters import CharacterTextSplitter

    client = ChromaService.client
    test_collection_name = "test_chroma_service_e2e"

    # load the document and split it into chunks
    loader = TextLoader(f"{mock_txt_file}")
    documents = loader.load()

    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # FIXME: We need to make embedding_function optional
    collection: chromadb.Collection = ChromaService.add_collection(test_collection_name)

    # load it into Chroma
    db = Chroma.from_documents(docs, embedding_function, collection_name=test_collection_name, client=client)

    # query it
    query = "What did the president say about Ketanji Brown Jackson"
    docs = db.similarity_search(query)

    assert (
        docs[0].page_content
        == "In state after state, new laws have been passed, not only to suppress the vote, but to subvert entire elections.\n\nWe cannot let this happen.\n\nTonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you're at it, pass the Disclose Act so Americans can know who is funding our elections.\n\nTonight, I'd like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer-an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.\n\nOne of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.\n\nAnd I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation's top legal minds, who will continue Justice Breyer's legacy of excellence."
    )


@pytest.mark.services()
# FIXME: This is a work in progress till I can incorporate this into the main codebase
@pytest.mark.slow()
@pytest.mark.integration()
@pytest.mark.e2e()
def test_chroma_service_e2e_add_to_chroma(
    mocker: MockerFixture, mock_txt_file: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    from goob_ai.services.chroma_service import ChromaService

    client = ChromaService.client
    test_collection_name = "test_chroma_service_e2e_add_to_chroma"

    db = ChromaService.add_to_chroma(
        path_to_document=f"{mock_txt_file}", collection_name=test_collection_name, embedding_function=None
    )

    # query it
    query = "What did the president say about Ketanji Brown Jackson"
    docs = db.similarity_search(query)

    assert (
        docs[0].page_content
        == "In state after state, new laws have been passed, not only to suppress the vote, but to subvert entire elections.\n\nWe cannot let this happen.\n\nTonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you're at it, pass the Disclose Act so Americans can know who is funding our elections.\n\nTonight, I'd like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer-an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.\n\nOne of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.\n\nAnd I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation's top legal minds, who will continue Justice Breyer's legacy of excellence."
    )

    logged_messages = [rec.message for rec in caplog.records]
    assert f"path_to_document = {mock_txt_file}" in logged_messages
    assert f"collection_name = {test_collection_name}" in logged_messages
    assert "embedding_function = None" in logged_messages


@pytest.mark.services()
# @pysnooper.snoop()
@pytest.mark.slow()
@pytest.mark.integration()
@pytest.mark.e2e()
def test_chroma_service_e2e_add_to_chroma_disallowed_special(mocker: MockerFixture, mock_txt_file: Path) -> None:
    from goob_ai.services.chroma_service import ChromaService

    client = ChromaService.client
    test_collection_name = "test_chroma_service_e2e_add_to_chroma_disallowed_special"

    embeddings = get_rag_embedding_function(filename=f"{mock_txt_file}", disallowed_special=())

    vectorstore: ChromaVectorStore = ChromaService.add_to_chroma(
        path_to_document=f"{mock_txt_file}", collection_name=test_collection_name, embedding_function=embeddings
    )

    # query it
    query = "What did the president say about Ketanji Brown Jackson"
    docs: list[Document] = vectorstore.similarity_search(query)

    assert (
        docs[0].page_content
        == "In state after state, new laws have been passed, not only to suppress the vote, but to subvert entire elections.\n\nWe cannot let this happen.\n\nTonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you're at it, pass the Disclose Act so Americans can know who is funding our elections.\n\nTonight, I'd like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer-an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.\n\nOne of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.\n\nAnd I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation's top legal minds, who will continue Justice Breyer's legacy of excellence."
    )


@pytest.mark.services()
# FIXME: This is a work in progress till I can incorporate this into the main codebase
@pytest.mark.slow()
@pytest.mark.integration()
@pytest.mark.e2e()
def test_chroma_service_e2e_add_to_chroma_url(mocker: MockerFixture) -> None:
    from goob_ai.services.chroma_service import ChromaService

    client = ChromaService.client
    test_collection_name = "test_chroma_service_e2e_add_to_chroma_url"

    db = ChromaService.add_to_chroma(
        path_to_document="https://lilianweng.github.io/posts/2023-06-23-agent/",
        collection_name=test_collection_name,
        embedding_function=None,
    )

    # query it
    query = "What is tool usage?"
    docs = db.similarity_search(query)

    assert (
        docs[0].page_content
        == "Fig. 9. Comparison of MIPS algorithms, measured in recall@10. (Image source: Google Blog, 2020)\nCheck more MIPS algorithms and performance comparison in ann-benchmarks.com.\nComponent Three: Tool Use#\nTool use is a remarkable and distinguishing characteristic of human beings. We create, modify and utilize external objects to do things that go beyond our physical and cognitive limits. Equipping LLMs with external tools can significantly extend the model capabilities."
    )


@pytest.mark.services()
@pytest.mark.integration()
@pytest.mark.parametrize(
    "filename, expected_suffix",
    [
        ("example.txt", ".txt"),
        ("document.pdf", ".pdf"),
        ("image.jpg", ".jpg"),
        ("file.docx", ".docx"),
        ("archive.tar.gz", ".gz"),
        ("no_extension", ""),
    ],
)
def test_get_suffix(filename: str, expected_suffix: str) -> None:
    """
    Test the get_suffix function.

    This test verifies that the `get_suffix` function correctly extracts
    the file extension from the given filename without the leading period.

    Args:
        filename (str): The filename to test.
        expected_suffix (str): The expected file extension without the leading period.
    """
    suffix = get_suffix(filename)
    assert suffix == expected_suffix


@pytest.mark.services()
@pytest.mark.integration()
def test_get_suffix_empty_filename() -> None:
    """
    Test the get_suffix function with an empty filename.

    This test verifies that the `get_suffix` function returns an empty string
    when given an empty filename.
    """
    filename = ""
    expected_suffix = ""
    suffix = get_suffix(filename)
    assert suffix == expected_suffix


@pytest.mark.services()
@pytest.mark.integration()
def test_get_suffix_multiple_dots() -> None:
    """
    Test the get_suffix function with a filename containing multiple dots.

    This test verifies that the `get_suffix` function correctly extracts
    the file extension from a filename containing multiple dots.
    """
    filename = "file.name.with.multiple.dots.txt"
    expected_suffix = ".txt"
    suffix = get_suffix(filename)
    assert suffix == expected_suffix


@pytest.mark.services()
@pytest.mark.integration()
@pytest.mark.parametrize(
    "filename, expected_result",
    [
        ("example.pdf", True),
        ("document.PDF", True),
        ("file.pdf", True),
        ("image.jpg", False),
        ("text.txt", False),
        ("archive.tar.gz", False),
        ("no_extension", False),
    ],
)
def test_is_pdf(filename: str, expected_result: bool) -> None:
    """
    Test the is_pdf function.

    This test verifies that the `is_pdf` function correctly determines
    whether a given filename has a PDF extension.

    Args:
        filename (str): The filename to test.
        expected_result (bool): The expected result (True if PDF, False otherwise).
    """
    result = is_pdf(filename)
    assert result == expected_result


@pytest.mark.services()
@pytest.mark.integration()
def test_is_pdf_empty_filename() -> None:
    """
    Test the is_pdf function with an empty filename.

    This test verifies that the `is_pdf` function returns False
    when given an empty filename.
    """
    filename = ""
    expected_result = False
    result = is_pdf(filename)
    assert result == expected_result


@pytest.mark.services()
@pytest.mark.integration()
def test_is_pdf_no_extension() -> None:
    """
    Test the is_pdf function with a filename without an extension.

    This test verifies that the `is_pdf` function returns False
    when given a filename without an extension.
    """
    filename = "file_without_extension"
    expected_result = False
    result = is_pdf(filename)
    assert result == expected_result


@pytest.mark.services()
@pytest.mark.integration()
@pytest.mark.parametrize(
    "filename, expected_result",
    [
        ("example.txt", True),
        ("document.TXT", True),
        ("file.txt", True),
        ("image.jpg", False),
        ("document.pdf", False),
        ("archive.tar.gz", False),
        ("no_extension", False),
    ],
)
def test_is_txt(filename: str, expected_result: bool) -> None:
    """
    Test the is_txt function.

    This test verifies that the `is_txt` function correctly determines
    whether a given filename has a TXT extension.

    Args:
        filename (str): The filename to test.
        expected_result (bool): The expected result (True if TXT, False otherwise).
    """
    result = is_txt(filename)
    assert result == expected_result


@pytest.mark.services()
@pytest.mark.integration()
def test_is_txt_empty_filename() -> None:
    """
    Test the is_txt function with an empty filename.

    This test verifies that the `is_txt` function returns False
    when given an empty filename.
    """
    filename = ""
    expected_result = False
    result = is_txt(filename)
    assert result == expected_result


@pytest.mark.services()
@pytest.mark.integration()
def test_is_txt_no_extension() -> None:
    """
    Test the is_txt function with a filename without an extension.

    This test verifies that the `is_txt` function returns False
    when given a filename without an extension.
    """
    filename = "file_without_extension"
    expected_result = False
    result = is_txt(filename)
    assert result == expected_result


@pytest.mark.services()
@pytest.mark.integration()
def test_get_rag_loader_real_pdf(mock_pdf_file: Path) -> None:
    """
    Test the get_rag_loader function.

    This test verifies that the `get_rag_loader` function returns the correct loader
    class based on the file extension or URL of the given document path.

    Args:
        path_to_document (str): The path or URL of the document.
        expected_loader_class (type | None): The expected loader class or None if no suitable loader is found.
    """
    loader_class = get_rag_loader(mock_pdf_file)
    assert "PyMuPDFLoader" in str(loader_class)
    # isinstance(loader_class, PyPDFLoader)


@pytest.mark.services()
@pytest.mark.integration()
def test_get_rag_loader_github_io_url(mock_github_io_url: FixtureRequest) -> None:
    """
    Test the get_rag_loader function for github.io urls.

    This test verifies that the `get_rag_loader` function returns the correct loader
    class based on the file extension or URL of the given document path.

    Args:
        path_to_document (str): The path or URL of the document.
        expected_loader_class (type | None): The expected loader class or None if no suitable loader is found.
    """
    loader_class = get_rag_loader(mock_github_io_url)
    assert "WebBaseLoader" in str(loader_class)


@pytest.mark.services()
@pytest.mark.integration()
def test_get_rag_loader_txt(mock_txt_file: FixtureRequest) -> None:
    """
    Test the get_rag_loader function for txt files.

    This test verifies that the `get_rag_loader` function returns the correct loader
    class based on the file extension or URL of the given document path.

    Args:
        path_to_document (str): The path or URL of the document.
        expected_loader_class (type | None): The expected loader class or None if no suitable loader is found.
    """
    loader_class = get_rag_loader(mock_txt_file)
    assert "TextLoader" in str(loader_class)


@pytest.mark.services()
@pytest.mark.integration()
def test_get_rag_loader_empty_path() -> None:
    """
    Test the get_rag_loader function with an empty path.

    This test verifies that the `get_rag_loader` function returns None
    when given an empty path.
    """
    path_to_document = ""
    expected_loader_class = None
    loader_class = get_rag_loader(path_to_document)
    assert loader_class == expected_loader_class


@pytest.mark.services()
@pytest.mark.integration()
def test_get_rag_loader_unsupported_extension() -> None:
    """
    Test the get_rag_loader function with an unsupported file extension.

    This test verifies that the `get_rag_loader` function returns None
    when given a file with an unsupported extension.
    """
    path_to_document = "file.unsupported"
    expected_loader_class = None
    loader_class = get_rag_loader(path_to_document)
    assert loader_class == expected_loader_class


@pytest.mark.services()
@pytest.mark.parametrize(
    "uri, expected_result",
    [
        ("https://example.com", True),
        ("http://subdomain.example.com/path", True),
        ("ftp://ftp.example.com", True),
        ("mailto:user@example.com", True),
        ("file:///path/to/file.txt", True),
        ("invalid_uri", False),
        ("http:/example.com", True),
        # ("https://", False), # FIXME: This is not a valid uri
        ("", False),
        ("http://example.com:8080/path?query=value#fragment", True),
        ("https://user:pass@example.com:8080/path", True),
    ],
)
def test_is_valid_uri(uri: str, expected_result: bool) -> None:
    """
    Test the is_valid_uri function with various URIs.

    This test verifies that the `is_valid_uri` function correctly determines
    whether a given URI is valid or not.

    Args:
        uri (str): The URI to test.
        expected_result (bool): The expected result (True if valid, False otherwise).
    """
    result = is_valid_uri(uri)
    assert result == expected_result


@pytest.mark.services()
def test_is_valid_uri_with_none() -> None:
    """
    Test the is_valid_uri function with None as input.

    This test verifies that the `is_valid_uri` function handles None input correctly.
    """
    with pytest.raises((AttributeError, TypeError)):
        is_valid_uri(None)


@pytest.mark.services()
def test_is_valid_uri_with_non_string() -> None:
    """
    Test the is_valid_uri function with a non-string input.

    This test verifies that the `is_valid_uri` function handles non-string input correctly.
    """
    with pytest.raises((AttributeError, TypeError)):
        is_valid_uri(123)


@pytest.mark.services()
@pytest.mark.parametrize(
    "filename, expected_result",
    [
        ("https://username.github.io", True),
        ("https://username.github.io/path/to/resource", True),
        ("http://username.github.io", True),
        ("https://username.github.io/", True),
        ("https://username.github.io/repo", True),
        ("https://username.github.io/repo/", True),
        ("https://username.github.io/repo/index.html", True),
        ("https://example.com", False),
        # ("http://example.github.io", False),
        ("https://username.github.com", False),
        # ("https://username.github.io/path/to/resource/", False),
        ("invalid_url", False),
        ("", False),
    ],
)
def test_is_github_io_url(filename: str, expected_result: bool) -> None:
    """
    Test the is_github_io_url function with various filenames.

    This test verifies that the `is_github_io_url` function correctly determines
    whether a given filename is a valid GitHub Pages URL.

    Args:
        filename (str): The filename to test.
        expected_result (bool): The expected result (True if valid GitHub Pages URL, False otherwise).
    """
    result = is_github_io_url(filename)
    assert result == expected_result


@pytest.mark.services()
@pytest.mark.integration()
def test_get_rag_splitter_txt(mock_txt_file: Path) -> None:
    """
    Test the get_rag_splitter function for txt files.

    This test verifies that the `get_rag_splitter` function returns the correct splitter
    class based on the file extension or URL of the given document path.

    Args:
        mock_txt_file (Path): The path to the mock txt file.
    """
    splitter_class = get_rag_splitter(str(mock_txt_file))
    assert "CharacterTextSplitter" in str(splitter_class)


@pytest.mark.services()
@pytest.mark.integration()
def test_get_rag_splitter_github_io_url(mock_github_io_url: str) -> None:
    """
    Test the get_rag_splitter function for github.io urls.

    This test verifies that the `get_rag_splitter` function returns the correct splitter
    class based on the file extension or URL of the given document path.

    Args:
        mock_github_io_url (str): The mock github.io url.
    """
    splitter_class = get_rag_splitter(mock_github_io_url)
    assert "RecursiveCharacterTextSplitter" in str(splitter_class)


@pytest.mark.services()
@pytest.mark.integration()
def test_get_rag_splitter_pdf(mock_pdf_file: Path) -> None:
    """
    Test the get_rag_splitter function for pdf files.

    This test verifies that the `get_rag_splitter` function returns None
    when given a pdf file path.

    Args:
        mock_pdf_file (Path): The path to the mock pdf file.
    """
    splitter_class = get_rag_splitter(str(mock_pdf_file))
    assert splitter_class is None


@pytest.mark.services()
@pytest.mark.integration()
def test_get_rag_splitter_empty_path() -> None:
    """
    Test the get_rag_splitter function with an empty path.

    This test verifies that the `get_rag_splitter` function returns None
    when given an empty path.
    """
    path_to_document = ""
    splitter_class = get_rag_splitter(path_to_document)
    assert splitter_class is None


@pytest.mark.services()
@pytest.mark.integration()
def test_get_rag_splitter_unsupported_extension() -> None:
    """
    Test the get_rag_splitter function with an unsupported file extension.

    This test verifies that the `get_rag_splitter` function returns None
    when given a file with an unsupported extension.
    """
    path_to_document = "file.unsupported"
    splitter_class = get_rag_splitter(path_to_document)
    assert splitter_class is None


@pytest.mark.services()
@pytest.mark.integration()
def test_get_rag_embedding_function_txt(mock_txt_file: Path) -> None:
    """
    Test the get_rag_embedding_function for txt files.

    This test verifies that the `get_rag_embedding_function` returns the correct embedding
    function based on the file extension or URL of the given document path.

    Args:
        mock_txt_file (Path): The path to the mock txt file.
    """
    embedding_function = get_rag_embedding_function(str(mock_txt_file))
    assert "langchain_community.embeddings.huggingface.HuggingFaceEmbeddings" in str(type(embedding_function))


@pytest.mark.services()
@pytest.mark.integration()
def test_get_rag_embedding_function_github_io_url(mock_github_io_url: str) -> None:
    """
    Test the get_rag_embedding_function for github.io urls.

    This test verifies that the `get_rag_embedding_function` returns the correct embedding
    function based on the file extension or URL of the given document path.

    Args:
        mock_github_io_url (str): The mock github.io url.
    """
    embedding_function = get_rag_embedding_function(mock_github_io_url)
    assert "langchain_openai.embeddings.base.OpenAIEmbeddings" in str(type(embedding_function))


@pytest.mark.services()
@pytest.mark.integration()
def test_get_rag_embedding_function_pdf(mock_pdf_file: Path) -> None:
    """
    Test the get_rag_embedding_function for pdf files.

    This test verifies that the `get_rag_embedding_function` returns the correct embedding
    function based on the file extension or URL of the given document path.

    Args:
        mock_pdf_file (Path): The path to the mock pdf file.
    """
    embedding_function = get_rag_embedding_function(str(mock_pdf_file))
    assert "openai.resources.embeddings.Embeddings" in str(embedding_function)


@pytest.mark.services()
@pytest.mark.integration()
def test_get_rag_embedding_function_empty_path() -> None:
    """
    Test the get_rag_embedding_function with an empty path.

    This test verifies that the `get_rag_embedding_function` returns None
    when given an empty path.
    """
    path_to_document = ""
    embedding_function = get_rag_embedding_function(path_to_document)
    assert embedding_function is None


@pytest.mark.services()
@pytest.mark.integration()
def test_get_rag_embedding_function_unsupported_extension() -> None:
    """
    Test the get_rag_embedding_function with an unsupported file extension.

    This test verifies that the `get_rag_embedding_function` returns None
    when given a file with an unsupported extension.
    """
    path_to_document = "file.unsupported"
    embedding_function = get_rag_embedding_function(path_to_document)
    assert embedding_function is None


@pytest.fixture()
def dummy_chroma_db(mocker) -> Chroma:
    """Fixture to create a mock Chroma database."""
    db = get_chroma_db()
    return db


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
@pytest.mark.services()
# @pytest.mark.vcr(match_on=["request_matcher"])
# @pytest.mark.vcr(ignore_localhost=False)
# @pytest.mark.vcr()
# @pytest.mark.vcr(allow_playback_repeats=True)
@pytest.mark.vcr(allow_playback_repeats=True, match_on=["request_matcher"], ignore_localhost=False)
def test_search_db_returns_relevant_documents(caplog: LogCaptureFixture, capsys: CaptureFixture, vcr):
    """
    Test that search_db returns relevant documents when found.

    This test verifies that the `search_db` function returns a list of
    relevant documents and their scores when a match is found in the database.
    """
    caplog.set_level(logging.DEBUG)

    from goob_ai.services.chroma_service import _await_server

    collection_name = "test_goobie"
    chroma_client = get_client()
    _await_server(chroma_client, attempts=10)

    collections = chroma_client.list_collections()

    # Clean up the collection if it exists
    if collection_name in collections:
        chroma_client.delete_collection(collection_name)
        chroma_client.create_collection(collection_name, get_or_create=True)
    else:
        chroma_client.create_collection(collection_name, get_or_create=True)

    # import bpdb

    # # bpdb.set_trace()
    # # db = dummy_chroma_db
    # db = Chroma(
    #     client=ChromaService.client,
    #     collection_name=collection_name,
    #     embedding_function=None,
    # )
    # results = search_db(db, "test query")
    # query_text = "test query"
    # expected_results = [
    #     (Document(page_content="doc1"), 0.8),
    #     (Document(page_content="doc2"), 0.7),
    # ]

    # FIXME: # assert results == expected_results

    # out, err = capsys.readouterr()
    # with capsys.disabled():
    #     import rich
    #     rich.inspect(vcr, all=True)

    assert vcr.play_count == 1
    # assert results == expected_results
    # wait for logging to finish runnnig
    # await LOGGER.complete()

    # caplog.clear()
    # dummy_chroma_db.similarity_search_with_relevance_scores.assert_called_once_with(query_text, k=3)


# def test_search_db_returns_none_when_no_relevant_documents(dummy_chroma_db):
#     """
#     Test that search_db returns None when no relevant documents are found.

#     This test verifies that the `search_db` function returns None when no
#     relevant documents are found in the database or the relevance score is
#     below the threshold.
#     """
#     query_text = "test query"
#     dummy_chroma_db.similarity_search_with_relevance_scores.return_value = []

#     results = search_db(dummy_chroma_db, query_text)

#     assert results is None
#     dummy_chroma_db.similarity_search_with_relevance_scores.assert_called_once_with(query_text, k=3)


# def test_search_db_returns_none_when_relevance_score_below_threshold(dummy_chroma_db):
#     """
#     Test that search_db returns None when relevance score is below threshold.

#     This test verifies that the `search_db` function returns None when the
#     relevance score of the top result is below the specified threshold of 0.7.
#     """
#     query_text = "test query"
#     mock_results = [
#         (Document(page_content="doc1"), 0.6),
#         (Document(page_content="doc2"), 0.5),
#     ]
#     dummy_chroma_db.similarity_search_with_relevance_scores.return_value = mock_results

#     results = search_db(dummy_chroma_db, query_text)

#     assert results is None
#     dummy_chroma_db.similarity_search_with_relevance_scores.assert_called_once_with(query_text, k=3)


@pytest.mark.services()
@pytest.mark.integration()
def test_generate_context_text() -> None:
    """
    Test the generate_context_text function.

    This test verifies that the `generate_context_text` function correctly generates
    the context text from the given search results.
    """
    from goob_ai.services.chroma_service import generate_context_text
    from langchain.schema import Document

    results = [
        (Document(page_content="doc1"), 0.8),
        (Document(page_content="doc2"), 0.7),
        (Document(page_content="doc3"), 0.6),
    ]

    expected_context_text = "doc1\n\n---\n\ndoc2\n\n---\n\ndoc3"
    context_text = generate_context_text(results)

    assert context_text == expected_context_text


@pytest.mark.services()
@pytest.mark.integration()
def test_generate_context_text_empty_results() -> None:
    """
    Test the generate_context_text function with empty search results.

    This test verifies that the `generate_context_text` function returns an empty string
    when given an empty list of search results.
    """
    from goob_ai.services.chroma_service import generate_context_text

    results: list[tuple[Document, float]] = []

    expected_context_text = ""
    context_text = generate_context_text(results)

    assert context_text == expected_context_text


@pytest.mark.services()
@pytest.mark.integration()
def test_generate_context_text_single_result() -> None:
    """
    Test the generate_context_text function with a single search result.

    This test verifies that the `generate_context_text` function correctly generates
    the context text when given a single search result.
    """
    from goob_ai.services.chroma_service import generate_context_text
    from langchain.schema import Document

    results = [(Document(page_content="doc1"), 0.8)]

    expected_context_text = "doc1"
    context_text = generate_context_text(results)

    assert context_text == expected_context_text


@pytest.mark.services()
@pytest.mark.integration()
def test_generate_prompt(capsys: CaptureFixture, caplog: LogCaptureFixture) -> None:
    """
    Test the generate_prompt function.

    This test verifies that the `generate_prompt` function correctly generates
    the prompt using the given context and question.
    """
    from goob_ai.services.chroma_service import generate_prompt

    context = "This is the context for the prompt."
    question = "What is the question?"
    # Human:
    # Answer the question based only on the following context:

    # This is the context for the prompt.

    # ---

    # Answer the question based on the above context: What is the question?

    # expected_prompt = 'Human: \nAnswer the question based only on the following context:\n\nThis is the context for the prompt.\n\n---\n\nAnswer the question based on the above context: What is the question?'
    expected_prompt = "Human: \nAnswer the question based only on the following context:\n\nThis is the context for the prompt.\n\n---\n\nAnswer the question based on the above context: What is the question?\n"

    prompt = generate_prompt(context, question)

    # out, err = capsys.readouterr()
    # with capsys.disabled():
    #     import rich
    #     # rich.inspect(vcr, all=True)
    #     rich.print("out, err:")
    #     rich.print(out)
    #     rich.print(err)
    #     rich.print("prompt:")
    #     rich.print(prompt)

    assert prompt == str(expected_prompt)


@pytest.mark.services()
@pytest.mark.integration()
def test_generate_prompt_empty_context() -> None:
    """
    Test the generate_prompt function with an empty context.

    This test verifies that the `generate_prompt` function correctly generates
    the prompt when the context is an empty string.
    """
    from goob_ai.services.chroma_service import generate_prompt

    context = ""
    question = "What is the question?"

    expected_prompt = "Human: \nAnswer the question based only on the following context:\n\n\n\n---\n\nAnswer the question based on the above context: What is the question?\n"

    prompt = generate_prompt(context, question)

    assert prompt == str(expected_prompt)


@pytest.mark.services()
@pytest.mark.integration()
def test_generate_prompt_empty_question() -> None:
    """
    Test the generate_prompt function with an empty question.

    This test verifies that the `generate_prompt` function correctly generates
    the prompt when the question is an empty string.
    """
    from goob_ai.services.chroma_service import generate_prompt

    context = "This is the context for the prompt."
    question = ""

    expected_prompt = "Human: \nAnswer the question based only on the following context:\n\nThis is the context for the prompt.\n\n---\n\nAnswer the question based on the above context: \n"

    prompt = generate_prompt(context, question)

    assert prompt == str(expected_prompt)


@pytest.mark.services()
@pytest.mark.integration()
def test_generate_prompt_multiline_context() -> None:
    """
    Test the generate_prompt function with a multiline context.

    This test verifies that the `generate_prompt` function correctly generates
    the prompt when the context contains multiple lines.
    """
    from goob_ai.services.chroma_service import generate_prompt

    context = "This is the first line of the context.\nThis is the second line of the context."
    question = "What is the question?"

    expected_prompt = "Human: \nAnswer the question based only on the following context:\n\nThis is the first line of the context.\nThis is the second line of the context.\n\n---\n\nAnswer the question based on the above context: What is the question?\n"

    prompt = generate_prompt(context, question)

    assert prompt == str(expected_prompt)


@pytest.mark.services()
@pytest.mark.integration()
@pytest.mark.asyncio()
async def test_string_to_doc():
    text = "This is a test string."
    doc = await string_to_doc(text)
    assert isinstance(doc, Document)
    assert doc.page_content == text


@pytest.mark.services()
@pytest.mark.integration()
@pytest.mark.asyncio()
async def test_markdown_to_documents():
    markdown_text = "# Heading 1\n\nThis is a paragraph.\n\n## Heading 2\n\nAnother paragraph."
    docs = [Document(page_content=markdown_text)]
    split_docs = await markdown_to_documents(docs)
    assert len(split_docs) >= 1
    assert all(isinstance(doc, Document) for doc in split_docs)


# @pytest.mark.integration()
# def test_franchise_metadata():
#     record = {"id": 1, "name": "Test Franchise"}
#     metadata = {"source": "test_source"}
#     updated_metadata = franchise_metadata(record, metadata)
#     assert updated_metadata["source"] == "API"
#     assert updated_metadata["franchise_id"] == 1
#     assert updated_metadata["franchise_name"] == "Test Franchise"

# @pytest.mark.integration()
# @pytest.mark.asyncio()
# async def test_json_to_docs():
#     json_data = json.dumps([{"id": 1, "text": "Test document 1"}, {"id": 2, "text": "Test document 2"}])
#     jq_schema = ".[]"
#     docs = await json_to_docs(json_data, jq_schema, franchise_metadata)
#     assert len(docs) == 2
#     assert all(isinstance(doc, Document) for doc in docs)
#     assert docs[0].page_content == "Test document 1"
#     assert docs[1].page_content == "Test document 2"


@pytest.mark.services()
@pytest.mark.integration()
@pytest.mark.asyncio()
async def test_generate_document_hashes():
    docs = [
        Document(page_content="Test document 1", metadata={"source": "test_source", "id": "1"}),
        Document(page_content="Test document 2", metadata={"source": "test_source", "id": "2"}),
    ]
    hashes = await generate_document_hashes(docs)
    assert len(hashes) == 2
    assert all(isinstance(hash, str) for hash in hashes)


# @pytest.mark.asyncio()
# async def test_create_chroma_db(mocker):
#     mocker.patch("goob_ai.services.chroma_service.rm_chroma_db", return_value=None)
#     mocker.patch("goob_ai.services.chroma_service.Chroma.from_documents", return_value=None)
#     docs = [Document(page_content="Test document")]
#     await create_chroma_db("test_collection", docs)
#     assert CHROMA_PATH_API.absolute().exists()

# @pytest.mark.asyncio()
# async def test_rm_chroma_db(mocker):
#     mocker.patch("shutil.rmtree", return_value=None)
#     mocker.patch("asyncio.sleep", return_value=None)
#     await rm_chroma_db()
#     assert not CHROMA_PATH_API.exists()


# # FIXME: This test is not working, racecondition eb tween ingesting data and writing to db.
# @pytest.mark.unittest()
# @pytest.mark.asyncio()
# async def test_add_and_query_unittest(mocker: MockerFixture):
#     # Define test inputs
#     collection_name = "test_collection_intgr"
#     question = "What is the meaning of life?"
#     reset = True

#     # Call the add_and_query function
#     result: VectorStoreRetriever = ChromaService.add_and_query(collection_name, question, reset)

#     # Assert that the result is an instance of VectorStoreRetriever
#     assert isinstance(result, VectorStoreRetriever)


@pytest.fixture()
def mock_chroma_db(mocker: MockerFixture):
    """Fixture to create a mock Chroma database."""
    mock_db = mocker.MagicMock()
    mock_db.get.return_value = {"ids": []}
    return mock_db


@pytest.fixture()
def mock_loader(mocker: MockerFixture):
    """Fixture to create a mock loader."""
    mock_loader = mocker.MagicMock()
    mock_loader.load.return_value = [Document(page_content="Test document")]
    return mock_loader


@pytest.fixture()
def mock_text_splitter(mocker: MockerFixture):
    """Fixture to create a mock text splitter."""
    mock_splitter = mocker.MagicMock()
    mock_splitter.split_documents.return_value = [Document(page_content="Test chunk")]
    return mock_splitter


# @pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
# @pytest.mark.flaky()
@pytest.mark.services()
@pytest.mark.unittest()
@pytest.mark.vcr(
    allow_playback_repeats=True, match_on=["method", "scheme", "port", "path", "query"], ignore_localhost=False
)
def test_add_to_chroma(
    mocker: MockerFixture,
    mock_chroma_db: MockerFixture,
    mock_loader: MockerFixture,
    mock_text_splitter: MockerFixture,
    mock_pdf_file: Path,
    caplog: LogCaptureFixture,
    capsys: CaptureFixture,
    vcr: Any,
):
    """
    Test adding new documents to the Chroma database.

    This test verifies that the `add_or_update_documents` function correctly adds
    new documents to the Chroma database when they don't exist.
    """
    caplog.set_level(logging.DEBUG)

    from goob_ai.services.chroma_service import ChromaService, _await_server, load_documents

    collection_name = "hugo_johnson"
    chroma_client = get_client()
    _await_server(chroma_client, attempts=10)

    collections = chroma_client.list_collections()

    # Clean up the collection if it exists
    if collection_name in collections:
        chroma_client.delete_collection(collection_name)
        chroma_client.create_collection(collection_name, get_or_create=True)

    # query it
    query = "What did the president say about Ketanji Brown Jackson"

    mock_get_rag_splitter = mocker.patch("goob_ai.services.chroma_service.get_rag_splitter")
    mock_get_chroma_db = mocker.patch("goob_ai.services.chroma_service.get_chroma_db")
    mock_get_rag_loader = mocker.patch("goob_ai.services.chroma_service.get_rag_loader")

    mock_get_chroma_db.return_value = mock_chroma_db
    mock_get_rag_loader.return_value = mock_loader
    mock_get_rag_splitter.return_value = mock_text_splitter

    documents = load_documents()
    chunks = split_text(documents)

    add_or_update_documents(chunks, collection_name=collection_name)

    mock_chroma_db.add_documents.assert_called_once()
    mock_chroma_db.get.assert_called_once_with(include=[])


@pytest.mark.services()
@pytest.mark.unittest()
def test_add_or_update_documents_new_documents(
    mocker: MockerFixture,
    mock_chroma_db: MockerFixture,
    mock_loader: MockerFixture,
    mock_text_splitter: MockerFixture,
    mock_pdf_file: Path,
    caplog: LogCaptureFixture,
    capsys: CaptureFixture,
):
    """
    Test adding new documents to the Chroma database.

    This test verifies that the `add_or_update_documents` function correctly adds
    new documents to the Chroma database when they don't exist.
    """
    from goob_ai.services.chroma_service import load_documents

    caplog.set_level(logging.DEBUG)

    mock_get_rag_splitter = mocker.patch("goob_ai.services.chroma_service.get_rag_splitter")
    mock_get_chroma_db = mocker.patch("goob_ai.services.chroma_service.get_chroma_db")
    mock_get_rag_loader = mocker.patch("goob_ai.services.chroma_service.get_rag_loader")
    mock_get_rag_embedding_function = mocker.patch("goob_ai.services.chroma_service.get_rag_embedding_function")

    mock_get_chroma_db.return_value = mock_chroma_db
    mock_get_rag_loader.return_value = mock_loader
    mock_get_rag_splitter.return_value = mock_text_splitter

    collection_name = "daisy_johnson"

    documents = load_documents()
    chunks = split_text(documents)

    add_or_update_documents(chunks, collection_name=collection_name)

    mock_chroma_db.add_documents.assert_called_once()
    mock_chroma_db.get.assert_called_once_with(include=[])


@pytest.mark.services()
@pytest.mark.unittest()
def test_add_or_update_documents_existing_documents(
    mocker: MockerFixture,
    mock_chroma_db: MockerFixture,
    mock_loader: MockerFixture,
    mock_text_splitter: MockerFixture,
    mock_pdf_file: Path,
    caplog: LogCaptureFixture,
    capsys: CaptureFixture,
):
    """
    Test adding existing documents to the Chroma database.

    This test verifies that the `add_or_update_documents` function correctly skips
    adding documents that already exist in the Chroma database.
    """
    from goob_ai.services.chroma_service import load_documents

    caplog.set_level(logging.DEBUG)

    mock_get_rag_splitter = mocker.patch("goob_ai.services.chroma_service.get_rag_splitter")
    mock_get_chroma_db = mocker.patch("goob_ai.services.chroma_service.get_chroma_db")
    mock_get_rag_loader = mocker.patch("goob_ai.services.chroma_service.get_rag_loader")

    mock_get_chroma_db.return_value = mock_chroma_db
    mock_get_rag_loader.return_value = mock_loader
    mock_get_rag_splitter.return_value = mock_text_splitter
    mock_chroma_db.get.return_value = {"ids": ["test_chunk_id"]}

    collection_name = generate_random_name()

    documents = load_documents()
    chunks = split_text(documents)

    add_or_update_documents(chunks, collection_name=collection_name)

    assert mock_chroma_db.add_documents.call_count == 1
    assert mock_chroma_db.add_documents.call_args.kwargs == {"ids": ["None:None:0", "None:None:1", "None:None:2"]}

    assert mock_chroma_db.add_documents.call_args.args == (
        [
            Document(metadata={"start_index": 0, "id": "None:None:0"}, page_content="Test document"),
            Document(metadata={"start_index": 0, "id": "None:None:1"}, page_content="Test document"),
            Document(metadata={"start_index": 0, "id": "None:None:2"}, page_content="Test document"),
        ],
    )

    calls = [
        mocker.call(
            [
                Document(metadata={"start_index": 0, "id": "None:None:0"}, page_content="Test document"),
                Document(metadata={"start_index": 0, "id": "None:None:1"}, page_content="Test document"),
                Document(metadata={"start_index": 0, "id": "None:None:2"}, page_content="Test document"),
            ],
            ids=["None:None:0", "None:None:1", "None:None:2"],
        )
    ]

    assert mock_chroma_db.add_documents.call_args_list == calls
