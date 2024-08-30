# pyright: reportMissingTypeStubs=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportUndefinedVariable=false
# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
from __future__ import annotations

import logging
import shutil

from pathlib import Path
from typing import TYPE_CHECKING, Any, List

import faiss

from goob_ai.services import (
    answer_question_from_context,
    bm25_retrieval,
    create_question_answer_from_context_chain,
    encode_from_string,
    encode_pdf,
    get_chunk_by_index,
    read_pdf_to_string,
    replace_t_with_space,
    retrieve_context_per_question,
    retrieve_with_context_overlap,
    show_context,
    split_text_to_chunks_with_indices,
    text_wrap,
)
from langchain.vectorstores import FAISS, VectorStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger as LOGGER
from rank_bm25 import BM25Okapi

import pytest


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture()
def mock_pdf_climate_change_file(tmp_path: Path) -> Path:
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
    test_pdf_path: Path = tmp_path / "Understanding_Climate_Change.pdf"
    shutil.copy("src/goob_ai/data/chroma/documents/Understanding_Climate_Change.pdf", test_pdf_path)
    return test_pdf_path


@pytest.fixture()
def mock_vector_store() -> VectorStore:
    def _mock_vector_store(path_to_pdf: Path, chunk_size: int = 400, chunk_overlap: int = 200) -> VectorStore:
        content = read_pdf_to_string(f"{path_to_pdf}")
        docs = split_text_to_chunks_with_indices(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        # This line creates a Faiss index using the IndexFlatL2 class. The dimension of the index is determined by the length of the embedding generated for the query "hello world".
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

        # This line creates an instance of the FAISS vector store, specifying the embedding function (embeddings), the Faiss index (index), an in-memory document store (InMemoryDocstore()), and an empty dictionary to map index IDs to document store IDs.
        vectorstore = FAISS.from_documents(
            docs, embeddings, index=index, docstore_cls=InMemoryDocstore(), index_to_docstore_id={}
        )
        return vectorstore

    return _mock_vector_store


@pytest.mark.integration()
@pytest.mark.services()
def test_replace_t_with_space() -> None:
    """
    Test the replace_t_with_space function.

    This test verifies that the function correctly replaces tab characters with spaces
    in the page content of each document.
    """
    input_docs: list[Document] = [
        Document(page_content="This\tis\ta\ttest"),
        Document(page_content="Another\tdocument\twith\ttabs"),
    ]
    expected_output: list[Document] = [
        Document(page_content="This is a test"),
        Document(page_content="Another document with tabs"),
    ]

    result: list[Document] = replace_t_with_space(input_docs)

    assert result == expected_output


@pytest.mark.integration()
@pytest.mark.services()
def test_text_wrap() -> None:
    """
    Test the text_wrap function.

    This test verifies that the function correctly wraps text to the specified width.
    """
    input_text: str = "This is a long text that should be wrapped to a specific width."
    expected_output: str = "This is a long text that\nshould be wrapped to a\nspecific width."

    result: str = text_wrap(input_text, width=25)

    assert result == expected_output


@pytest.mark.integration()
@pytest.mark.services()
def test_encode_pdf(mock_pdf_climate_change_file: Path) -> None:
    """
    Test the encode_pdf function.

    This test verifies that the function correctly encodes a PDF file into a FAISS vector store.

    Args:
    ----
        mock_pdf_climate_change_file (Path): The path to the mock PDF file.
    """
    result: FAISS = encode_pdf(str(mock_pdf_climate_change_file))

    assert isinstance(result, FAISS)
    assert len(result.index_to_docstore_id) > 0


@pytest.mark.integration()
@pytest.mark.services()
def test_encode_from_string() -> None:
    """
    Test the encode_from_string function.

    This test verifies that the function correctly encodes a string into a FAISS vector store.
    """
    input_content: str = "This is a test content to be encoded into a vector store."

    result: FAISS = encode_from_string(input_content)

    assert isinstance(result, FAISS)
    assert len(result.index_to_docstore_id) > 0


@pytest.mark.integration()
@pytest.mark.services()
def test_retrieve_context_per_question(mocker: MockerFixture) -> None:
    """
    Test the retrieve_context_per_question function.

    This test verifies that the function correctly retrieves relevant context for a given question.

    Args:
    ----
        mocker (MockerFixture): Pytest mocker fixture.
    """
    mock_retriever = mocker.Mock()
    mock_retriever.get_relevant_documents.return_value = [
        Document(page_content="Relevant context 1"),
        Document(page_content="Relevant context 2"),
    ]

    question: str = "What is the meaning of life?"
    result: list[str] = retrieve_context_per_question(question, mock_retriever)

    assert result == ["Relevant context 1", "Relevant context 2"]
    mock_retriever.get_relevant_documents.assert_called_once_with(question)


@pytest.mark.integration()
@pytest.mark.services()
def test_create_question_answer_from_context_chain() -> None:
    """
    Test the create_question_answer_from_context_chain function.

    This test verifies that the function correctly creates a chain for answering questions based on context.
    """
    llm: ChatOpenAI = ChatOpenAI()

    result: RunnableSerializable = create_question_answer_from_context_chain(llm)

    assert isinstance(result, RunnableSerializable)
    # assert isinstance(result.prompt, PromptTemplate)


@pytest.mark.integration()
@pytest.mark.services()
def test_answer_question_from_context(mocker: MockerFixture) -> None:
    """
    Test the answer_question_from_context function.

    This test verifies that the function correctly answers a question using the given context.

    Args:
    ----
        mocker (MockerFixture): Pytest mocker fixture.
    """
    mock_chain = mocker.Mock()
    mock_chain.invoke.return_value.answer_based_on_content = "This is the answer."

    question: str = "What is the question?"
    context: list[str] = ["Context 1", "Context 2"]

    result: dict = answer_question_from_context(question, context, mock_chain)

    assert result == {
        "answer": "This is the answer.",
        "context": ["Context 1", "Context 2"],
        "question": "What is the question?",
    }
    mock_chain.invoke.assert_called_once_with({"question": question, "context": context})


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
@pytest.mark.integration()
@pytest.mark.services()
def test_show_context(caplog: LogCaptureFixture) -> None:
    """
    Test the show_context function.

    This test verifies that the function correctly logs the contents of the provided context list.

    Args:
    ----
        caplog (LogCaptureFixture): Pytest fixture to capture log output.
    """
    caplog.set_level(logging.INFO)
    context: list[str] = ["Context 1", "Context 2"]

    show_context(context)

    test_logs = [i.message for i in caplog.records if i.levelno == logging.INFO]

    assert "Context 1:" in test_logs
    assert "Context 1" in test_logs
    assert "Context 2:" in test_logs
    assert "Context 2" in test_logs


@pytest.mark.integration()
@pytest.mark.services()
def test_read_pdf_to_string(mock_pdf_climate_change_file: Path) -> None:
    """
    Test the read_pdf_to_string function.

    This test verifies that the function correctly reads a PDF document and returns its content as a string.

    Args:
    ----
        mock_pdf_climate_change_file (Path): The path to the mock PDF file.
    """
    result: str = read_pdf_to_string(str(mock_pdf_climate_change_file))

    assert isinstance(result, str)
    assert len(result) > 0
    assert "climate change" in result.lower()


@pytest.mark.integration()
@pytest.mark.services()
def test_bm25_retrieval() -> None:
    """
    Test the bm25_retrieval function.

    This test verifies that the function correctly performs BM25 retrieval and returns the top k cleaned text chunks.
    """
    cleaned_texts: list[str] = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]
    bm25: BM25Okapi = BM25Okapi([text.split() for text in cleaned_texts])
    query: str = "first document"

    result: list[str] = bm25_retrieval(bm25, cleaned_texts, query, k=2)

    assert len(result) == 2
    assert "this document is the second document." in result[0].lower()


# FIXME: fix all of the tests below
@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
@pytest.mark.integration()
@pytest.mark.services()
def test_split_text_to_chunks_with_indices() -> None:
    """
    Test the split_text_to_chunks_with_indices function.

    This test verifies that the function correctly splits text into chunks with metadata about the chunk's index.
    """
    text: str = "This is a sample text. It will be split into chunks. Each chunk will have metadata about its index."
    chunk_size: int = 20
    chunk_overlap: int = 5

    result: list[dict[str, Document]] = split_text_to_chunks_with_indices(text, chunk_size, chunk_overlap)

    assert len(result) == 7
    assert (
        result[0].metadata["text"]
        == "This is a sample text. It will be split into chunks. Each chunk will have metadata about its index."
    )
    assert result[0].metadata["index"] == 0

    assert (
        result[1].metadata["text"]
        == "This is a sample text. It will be split into chunks. Each chunk will have metadata about its index."
    )
    assert result[1].metadata["index"] == 1


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
@pytest.mark.integration()
@pytest.mark.services()
def test_get_chunk_by_index(mock_vector_store: VectorStore, mock_pdf_climate_change_file: Path) -> None:
    """
    Test the get_chunk_by_index function.

    This test verifies that the function correctly retrieves a chunk from a vector store based on its index.

    Args:
    ----
        mock_vector_store (VectorStore): A mock vector store containing chunks with metadata.
    """
    vectorstore = mock_vector_store(mock_pdf_climate_change_file, chunk_size=400, chunk_overlap=200)
    index: int = 1

    result: Document = get_chunk_by_index(vectorstore, index)

    assert isinstance(result, Document)
    assert result.metadata["chunk_index"] == index


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
@pytest.mark.integration()
@pytest.mark.services()
def test_retrieve_with_context_overlap(mock_vector_store: VectorStore, mock_pdf_climate_change_file: Path) -> None:
    """
    Test the retrieve_with_context_overlap function.

    This test verifies that the function correctly retrieves chunks with context overlap based on a query.

    Args:
    ----
        mock_vector_store (VectorStore): A mock vector store containing chunks with metadata.
    """
    vectorstore = mock_vector_store(mock_pdf_climate_change_file, chunk_size=400, chunk_overlap=200)
    query: str = "sample query"
    k: int = 3
    chunk_overlap: int = 5

    result: list[str] = retrieve_with_context_overlap(vectorstore, query, k, chunk_overlap)

    assert len(result) == k
    for chunk in result:
        assert isinstance(chunk, str)
        assert len(chunk) > 0

    # vector_store.delete(ids=[uuids[-1]])
