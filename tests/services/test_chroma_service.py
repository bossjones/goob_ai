from __future__ import annotations

import os
import shutil

from pathlib import Path
from typing import TYPE_CHECKING, Generator, Iterable, Iterator

from goob_ai.aio_settings import aiosettings
from goob_ai.services.chroma_service import CustomOpenAIEmbeddings, generate_data_store, get_response, save_to_chroma
from langchain.schema import Document

import pytest


if TYPE_CHECKING:
    from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock

    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_openai_api_key(mocker: MockerFixture) -> str:
    """Fixture to provide a mock OpenAI API key for testing purposes.

    This fixture returns a mock OpenAI API key that can be used in tests
    to simulate the presence of a valid API key without making actual
    API calls.

    Args:
        mocker (MockerFixture): The mocker fixture for patching.

    Returns:
        str: A mock OpenAI API key.
    """
    return "test_api_key"


@pytest.fixture
def custom_embeddings(mock_openai_api_key: str) -> CustomOpenAIEmbeddings:
    """Create a CustomOpenAIEmbeddings instance with the provided API key.

    Args:
        mock_openai_api_key (str): The OpenAI API key to use for the embeddings.

    Returns:
        CustomOpenAIEmbeddings: An instance of CustomOpenAIEmbeddings initialized with the provided API key.
    """
    return CustomOpenAIEmbeddings(openai_api_key=mock_openai_api_key)


def test_custom_openai_embeddings_init(mocker: MockerFixture, monkeypatch: MonkeyPatch) -> None:
    """
    Test the initialization of CustomOpenAIEmbeddings.

    This test verifies that the CustomOpenAIEmbeddings instance is initialized
    with the correct OpenAI API key.

    Args:
        mocker (MockerFixture): The mocker fixture for patching.
    """
    mock_openai_api_key = "test_api_key"
    monkeypatch.setattr(aiosettings, "openai_api_key", mock_openai_api_key)

    embeddings = CustomOpenAIEmbeddings(openai_api_key=mock_openai_api_key)
    assert embeddings.openai_api_key == "test_api_key"


@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("DEBUG_AIDER"),
    reason="These tests are meant to only run locally on laptop prior to porting it over to new system",
)
def test_custom_openai_embeddings_call(mocker: MockerFixture, custom_embeddings: CustomOpenAIEmbeddings) -> None:
    """Test the call method of CustomOpenAIEmbeddings.

    This test verifies that the call method of CustomOpenAIEmbeddings returns
    the expected embeddings for the given texts.

    Args:
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


@pytest.fixture
def mock_pdf_file(tmp_path: Path) -> Path:
    """Fixture to create a mock PDF file for testing purposes.

    This fixture creates a temporary directory and copies a test PDF file into it.
    The path to the mock PDF file is then returned for use in tests.

    Args:
        tmp_path (Path): The temporary path provided by pytest.

    Returns:
        Path: A Path object of the path to the mock PDF file.
    """
    test_pdf_path: Path = tmp_path / "rich-readthedocs-io-en-latest.pdf"
    shutil.copy("src/goob_ai/data/chroma/documents/rich-readthedocs-io-en-latest.pdf", test_pdf_path)
    return test_pdf_path


def test_load_documents(mocker: MockerFixture, mock_pdf_file: Path) -> None:
    """Test the loading of documents from a PDF file.

    This test verifies that the `load_documents` function correctly loads
    documents from a PDF file, splits the text into chunks, and saves the
    chunks to Chroma.

    Args:
        mocker (MockerFixture): The mocker fixture for patching.
        mock_pdf_file (Path): The path to the mock PDF file.

    The test performs the following steps:
    1. Mocks the `os.listdir` and `os.path.join` functions to simulate the presence of the PDF file.
    2. Mocks the `PyPDFLoader` to return a document with test content.
    3. Calls the `generate_data_store` function to load, split, and save the document.
    4. Asserts that the document is loaded, split, and saved correctly.
    """
    mocker.patch("os.listdir", return_value=["rich-readthedocs-io-en-latest.pdf"])
    mocker.patch("os.path.join", return_value=mock_pdf_file)
    mock_loader = mocker.patch("goob_ai.services.chroma_service.PyPDFLoader")
    mock_loader.return_value.load.return_value = [Document(page_content="Test content", metadata={})]

    from goob_ai.services.chroma_service import load_documents

    documents = load_documents()

    assert len(documents) == 1
    assert documents[0].page_content == "Test content"
    mock_loader.return_value.load.assert_called_once_with()
    mock_load_documents = mocker.patch(
        "goob_ai.services.chroma_service.load_documents",
        return_value=[Document(page_content="Test content", metadata={})],
    )
    mock_split_text = mocker.patch(
        "goob_ai.services.chroma_service.split_text", return_value=[Document(page_content="Test chunk", metadata={})]
    )
    mock_save_to_chroma: MagicMock | AsyncMock | NonCallableMagicMock = mocker.patch(
        "goob_ai.services.chroma_service.save_to_chroma"
    )

    generate_data_store()

    mock_load_documents.assert_called_once()
    mock_split_text.assert_called_once_with([Document(page_content="Test content", metadata={})])
    mock_save_to_chroma.assert_called_once_with([Document(page_content="Test chunk", metadata={})])


@pytest.mark.slow
# @pytest.mark.skipif(
#     os.getenv("PINECONE_ENV"),
#     reason="These tests are meant to only run locally on laptop prior to porting it over to new system",
# )
def test_split_text(mocker: MockerFixture) -> None:
    """Test the split_text function.

    This test verifies that the `split_text` function correctly splits
    documents into chunks using the RecursiveCharacterTextSplitter.

    Args:
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

    mock_documents: List[Document] = [Document(page_content="This is a test document.", metadata={})]
    mock_chunks: List[Document] = [
        Document(page_content="This is a test", metadata={"start_index": 0}),
        Document(page_content="document.", metadata={"start_index": 15}),
    ]

    mock_text_splitter: MagicMock | AsyncMock | NonCallableMagicMock = mocker.patch(
        "goob_ai.services.chroma_service.RecursiveCharacterTextSplitter"
    )
    mock_text_splitter.return_value.split_documents.return_value = mock_chunks

    chunks: List[Document] = split_text(mock_documents)

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
