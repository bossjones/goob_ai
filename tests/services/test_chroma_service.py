import pytest
from pytest_mock import MockerFixture
from goob_ai.services.chroma_service import CustomOpenAIEmbeddings, generate_data_store
from goob_ai.aio_settings import aiosettings
from langchain.schema import Document

@pytest.fixture
def mock_openai_api_key(mocker):
    return "test_api_key"

@pytest.fixture
def custom_embeddings(mock_openai_api_key):
    return CustomOpenAIEmbeddings(openai_api_key=mock_openai_api_key)

def test_custom_openai_embeddings_init(mocker):
    mock_openai_api_key = "test_api_key"
    mocker.patch.object(aiosettings, 'openai_api_key', mock_openai_api_key)
    
    embeddings = CustomOpenAIEmbeddings()
    assert embeddings.openai_api_key == mock_openai_api_key

def test_custom_openai_embeddings_call(mocker, custom_embeddings):
    mock_texts = ["This is a test document."]
    mock_embeddings = [[0.1, 0.2, 0.3]]

    mocker.patch.object(custom_embeddings, '_embed_documents', return_value=mock_embeddings)

    result = custom_embeddings(mock_texts)
    assert result == mock_embeddings

def test_generate_data_store(mocker: MockerFixture):
    mock_load_documents = mocker.patch("goob_ai.services.chroma_service.load_documents", return_value=[Document(page_content="Test content", metadata={})])
    mock_split_text = mocker.patch("goob_ai.services.chroma_service.split_text", return_value=[Document(page_content="Test chunk", metadata={})])
    mock_save_to_chroma = mocker.patch("goob_ai.services.chroma_service.save_to_chroma")

    generate_data_store()

    mock_load_documents.assert_called_once()
    mock_split_text.assert_called_once_with([Document(page_content="Test content", metadata={})])
    mock_save_to_chroma.assert_called_once_with([Document(page_content="Test chunk", metadata={})])

def test_embed_documents(mocker, custom_embeddings):
    mock_texts = ["This is a test document."]
    mock_embeddings = [[0.1, 0.2, 0.3]]

    mocker.patch.object(custom_embeddings, 'embed_documents', return_value=mock_embeddings)

    result = custom_embeddings._embed_documents(mock_texts)
    assert result == mock_embeddings
