import pytest
from pytest_mock import MockerFixture
from goob_ai.services.chroma_service import CustomOpenAIEmbeddings, generate_data_store, get_response
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

def test_get_response(mocker: MockerFixture):
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

    mocker.patch.object(custom_embeddings, 'embed_documents', return_value=mock_embeddings)

    result = custom_embeddings._embed_documents(mock_texts)
    assert result == mock_embeddings
