from __future__ import annotations

import os
import shutil

from collections.abc import Generator, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from chromadb import Collection
from goob_ai.aio_settings import aiosettings
from goob_ai.services.chroma_service import (
    CustomOpenAIEmbeddings,
    generate_data_store,
    get_file_extension,
    get_response,
    get_suffix,
    save_to_chroma,
)
from langchain.schema import Document

import pytest


if TYPE_CHECKING:
    from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock

    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


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


def test_load_documents(mocker: MockerFixture, mock_pdf_file: Path) -> None:
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


# FIXME: This is a work in progress till I can incorporate this into the main codebase
@pytest.mark.slow()
@pytest.mark.integration()
@pytest.mark.e2e()
def test_chroma_service_e2e_add_to_chroma(mocker: MockerFixture, mock_txt_file: Path) -> None:
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
