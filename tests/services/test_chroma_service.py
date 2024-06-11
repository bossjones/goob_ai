from __future__ import annotations

import shutil


from pathlib import Path
from typing import Generator

from goob_ai.aio_settings import aiosettings
from goob_ai.services.chroma_service import CustomOpenAIEmbeddings, generate_data_store, get_response, save_to_chroma
from langchain.schema import Document

import pytest

from pytest_mock import MockerFixture


@pytest.fixture
def mock_openai_api_key(mocker: MockerFixture) -> str:
    return "test_api_key"


@pytest.fixture
def custom_embeddings(mock_openai_api_key: str) -> CustomOpenAIEmbeddings:
    return CustomOpenAIEmbeddings(openai_api_key=mock_openai_api_key)


def test_custom_openai_embeddings_init(mocker: MockerFixture) -> None:
    mock_openai_api_key = "test_api_key"
    mocker.patch.object(aiosettings, "openai_api_key", mock_openai_api_key)

    embeddings = CustomOpenAIEmbeddings()
    assert embeddings.openai_api_key == mock_openai_api_key


def test_custom_openai_embeddings_call(mocker: MockerFixture, custom_embeddings: CustomOpenAIEmbeddings) -> None:
    mock_texts: list[str] = ["This is a test document."]
    mock_embeddings: list[list[float]] = [[0.1, 0.2, 0.3]]

    mocker.patch.object(custom_embeddings, "_embed_documents", return_value=mock_embeddings)

    result: list[list[float]] = custom_embeddings(mock_texts)
    assert result == mock_embeddings
    mock_texts = ["This is a test document."]
    mock_embeddings = [[0.1, 0.2, 0.3]]

    mocker.patch.object(custom_embeddings, "_embed_documents", return_value=mock_embeddings)

    result = custom_embeddings(mock_texts)
    assert result == mock_embeddings



@pytest.fixture
def mock_pdf_file(tmp_path: Path) -> Generator[Path, None, None]:
    # Create a temporary directory and copy the test PDF file into it
    test_pdf_path = tmp_path / "rich-readthedocs-io-en-latest.pdf"
    shutil.copy("src/goob_ai/data/chroma/documents/rich-readthedocs-io-en-latest.pdf", test_pdf_path)
    return test_pdf_path


def test_load_documents(mocker: MockerFixture, mock_pdf_file: Path) -> None:
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
    mock_save_to_chroma = mocker.patch("goob_ai.services.chroma_service.save_to_chroma")

    generate_data_store()

    mock_load_documents.assert_called_once()
    mock_split_text.assert_called_once_with([Document(page_content="Test content", metadata={})])
    mock_save_to_chroma.assert_called_once_with([Document(page_content="Test chunk", metadata={})])


def test_split_text(mocker: MockerFixture) -> None:
    mock_documents = [Document(page_content="This is a test document.", metadata={})]
    mock_chunks = [
        Document(page_content="This is a test", metadata={"start_index": 0}),
        Document(page_content="document.", metadata={"start_index": 15}),
    ]

    mock_text_splitter = mocker.patch("goob_ai.services.chroma_service.RecursiveCharacterTextSplitter")
    mock_text_splitter.return_value.split_documents.return_value = mock_chunks

    from goob_ai.services.chroma_service import split_text

    chunks = split_text(mock_documents)

    assert len(chunks) == 2
    assert chunks[0].page_content == "This is a test"
    assert chunks[1].page_content == "document."
    mock_text_splitter.return_value.split_documents.assert_called_once_with(mock_documents)
    mock_generate_data_store = mocker.patch("goob_ai.services.chroma_service.generate_data_store")
    from goob_ai.services.chroma_service import main

    main()
    mock_generate_data_store.assert_called_once()
    mock_query_text = "What is the capital of France?"
    mock_results = [
        (Document(page_content="Paris is the capital of France.", metadata={"source": "source1"}), 0.9),
        (Document(page_content="France's capital is Paris.", metadata={"source": "source2"}), 0.85),
        (Document(page_content="The capital city of France is Paris.", metadata={"source": "source3"}), 0.8),
    ]
    mock_context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in mock_results])
    mock_prompt = f"""
Answer the question based only on the following context:

{mock_context_text}

---

Answer the question based on the above context: {mock_query_text}
"""
    mock_response_text = "The capital of France is Paris."
    mock_sources = ["source1", "source2", "source3"]

    mock_embedding_function = mocker.patch("goob_ai.services.chroma_service.OpenAIEmbeddings")
    mock_chroma = mocker.patch("goob_ai.services.chroma_service.Chroma")
    mock_chroma.return_value.similarity_search_with_relevance_scores.return_value = mock_results
    mock_prompt_template = mocker.patch("goob_ai.services.chroma_service.ChatPromptTemplate")
    mock_prompt_template.from_template.return_value.format.return_value = mock_prompt
    mock_model = mocker.patch("goob_ai.services.chroma_service.ChatOpenAI")
    mock_model.return_value.predict.return_value = mock_response_text

    response = get_response(mock_query_text)

    assert response == f"Response: {mock_response_text}\nSources: {mock_sources}"
    mock_chroma.return_value.similarity_search_with_relevance_scores.assert_called_once_with(mock_query_text, k=3)
    mock_prompt_template.from_template.assert_called_once_with("""
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
""")
    mock_model.return_value.predict.assert_called_once_with(mock_prompt)
    mock_texts = ["This is a test document."]
    mock_embeddings = [[0.1, 0.2, 0.3]]

    mocker.patch.object(custom_embeddings, "embed_documents", return_value=mock_embeddings)

    result = custom_embeddings._embed_documents(mock_texts)
    assert result == mock_embeddings
