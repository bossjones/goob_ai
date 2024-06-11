import pytest
from goob_ai.services.chroma_service import CustomOpenAIEmbeddings
from goob_ai.aio_settings import aiosettings

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

@pytest.fixture
def mock_openai_api_key(mocker):
    return "test_api_key"

@pytest.fixture
def custom_embeddings(mock_openai_api_key):
    return CustomOpenAIEmbeddings(openai_api_key=mock_openai_api_key)

def test_custom_openai_embeddings_call(mocker, custom_embeddings):
    mock_texts = ["This is a test document."]
    mock_embeddings = [[0.1, 0.2, 0.3]]

    mocker.patch.object(custom_embeddings, '_embed_documents', return_value=mock_embeddings)

    result = custom_embeddings(mock_texts)
    assert result == mock_embeddings

def test_embed_documents(mocker, custom_embeddings):
    mock_texts = ["This is a test document."]
    mock_embeddings = [[0.1, 0.2, 0.3]]

    mocker.patch.object(custom_embeddings, 'embed_documents', return_value=mock_embeddings)

    result = custom_embeddings._embed_documents(mock_texts)
    assert result == mock_embeddings
