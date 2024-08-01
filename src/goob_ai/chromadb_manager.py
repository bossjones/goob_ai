# pyright: reportCallInDefaultInitializer=false
# pylint: disable=no-member
# SOURCE: https://github.com/bhuvan454/bareRAG/tree/master
from __future__ import annotations

import os

from typing import Any, Dict, List, Optional, Tuple

import chromadb

from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.llms.ollama import Ollama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger as LOGGER
from tqdm import tqdm

from goob_ai.aio_settings import aiosettings


HERE = os.path.dirname(__file__)
QDRANT_URL = os.getenv("QDRANT_URL", "https://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "ragcollection")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
DATA_PATH = os.path.join(HERE, "data", "chroma", "data")
CHROMA_PATH = os.path.join(HERE, "data", "chroma", "vectorstorage")

# DATA_PATH = "/Users/malcolm/dev/bossjones/goob_ai/src/goob_ai/data/db"
# DATA_DOC_PATH = os.getenv("DATA_DOC_PATH", os.path.join(DATA_PATH, "documents"))
# CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(DATA_PATH, "chroma_db"))


PROMPT_TEMPLATE = """
Answer the question based only basedon the on the following context:

{context}

----------------------------------------

Answer the question based on the above context: {question}, and if you are not sure
about the answer, give a relevant answer based on the context and give certainty score of 0.5

"""


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks.

    Args:
        documents: List of documents to split.

    Returns:
        List of split document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    """Custom OpenAI embeddings class."""

    def __init__(self, openai_api_key: str = aiosettings.openai_api_key.get_secret_value()):
        """Initialize the CustomOpenAIEmbeddings class.

        Args:
            openai_api_key: OpenAI API key.
        """
        super().__init__(openai_api_key=openai_api_key)

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents using OpenAI embeddings.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings for each text.
        """
        return super().embed_documents(texts)

    def __call__(self, input: str) -> list[float]:
        """Call the embeddings on a single input text.

        Args:
            input: Input text to embed.

        Returns:
            Embedding for the input text.
        """
        return self._embed_documents([input])[0]


class QueryRAG:
    """Query RAG model."""

    def __init__(self, model: str):
        """Initialize the QueryRAG class.

        Args:
            model: Name of the model to use.
        """
        self.model = Ollama(model=model)

    def invoke(self, query_results: dict[str, Any], prompt: str) -> str:
        """Invoke the RAG model.

        Args:
            query_results: Query results from the database.
            prompt: Prompt for the model.

        Returns:
            Response from the model.
        """
        response = self.model.invoke(prompt)
        sources = [doc.get("chunk_id") for doc in query_results["metadatas"][0]]
        return f"Response: {response}\n\nSources: {sources}"


def load_pdf(file_path: str) -> list[Document]:
    """Load the PDF file and return the text content.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of documents containing the text content.
    """
    loader = PyMuPDFLoader(file_path)
    return loader.load()


def query_database(collection: chromadb.Collection, query_text: str) -> tuple[dict[str, Any], str]:
    """Query the database for relevant documents.

    Args:
        collection: Chroma collection to query.
        query_text: Query text.

    Returns:
        Tuple containing the query results and the prompt.
    """
    results = collection.query(query_texts=[query_text], n_results=int(RAG_TOP_K), include=["documents", "metadatas"])
    context_level = results["documents"][0]
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_level, question=query_text)
    return results, prompt


class ChromaDB:
    """Chroma database manager."""

    def __init__(self):
        """Initialize the ChromaDB class."""
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.doc_add_batch_size = 100

    def add_collection(self, collection_name: str, embedding_function: Any) -> chromadb.Collection:
        """Add a new collection to the database.

        Args:
            collection_name: Name of the collection.
            embedding_function: Embedding function to use.

        Returns:
            The created collection.
        """
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_function
        )
        return self.collection

    def get_list_collections(self) -> list[str]:
        """Get a list of all collections in the database.

        Returns:
            List of collection names.
        """
        return self.chroma_client.list_collections()

    def get_collection(self, collection_name: str, embedding_function: Any) -> Optional[chromadb.Collection]:
        """Get a collection by name.

        Args:
            collection_name: Name of the collection.
            embedding_function: Embedding function to use.

        Returns:
            The collection if found, None otherwise.
        """
        return self.chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)

    def add_chunk_ids(self, chunks: list[Document]) -> list[Document]:
        """Add chunk IDs to the document chunks.

        Args:
            chunks: List of document chunks.

        Returns:
            List of document chunks with chunk IDs added.
        """
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            chunk.metadata["chunk_id"] = chunk_id

        return chunks

    def add_chunks(self, chunks: list[Document]) -> None:
        """Add document chunks to the collection.

        Args:
            chunks: List of document chunks to add.
        """
        chunks_with_ids = self.add_chunk_ids(chunks)
        existing_items = self.collection.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing items in DB: {len(existing_ids)}")

        new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["chunk_id"] not in existing_ids]

        if len(new_chunks):
            for i in tqdm(range(0, len(new_chunks), self.doc_add_batch_size), desc="Adding new items to DB"):
                batch = new_chunks[i : i + self.doc_add_batch_size]

                batch_ids = [chunk.metadata["chunk_id"] for chunk in batch]
                documents = [chunk.page_content for chunk in batch]
                metadata = [chunk.metadata for chunk in batch]

                self.collection.upsert(documents=documents, ids=batch_ids, metadatas=metadata)
        else:
            print("No new items to add to the DB")

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the database.

        Args:
            collection_name: Name of the collection to delete.
        """
        self.chroma_client.delete_collection(collection_name)
        print("Collection deleted")


class ChromaDBManager:
    """Chroma database manager."""

    def __init__(self):
        """Initialize the ChromaDBManager class."""
        self.vector_db = ChromaDB()
        self.collection = None
        self.model = QueryRAG("llama3")

    def get_list_collections(self) -> list[str]:
        """Get a list of all collections in the database.

        Returns:
            List of collection names.
        """
        return self.vector_db.get_list_collections()

    def get_collection(self, collection_name: str, embedding_function: Any) -> Optional[chromadb.Collection]:
        """Get a collection by name.

        Args:
            collection_name: Name of the collection.
            embedding_function: Embedding function to use.

        Returns:
            The collection if found, None otherwise.
        """
        return self.vector_db.get_collection(
            collection_name, CustomOpenAIEmbeddings(openai_api_key=aiosettings.openai_api_key.get_secret_value())
        )

    def initialize_database(self, pdf_path: str, collection_name: str, debug: bool = False) -> None:
        """Initialize the database with data from a PDF file.

        Args:
            pdf_path: Path to the PDF file.
            collection_name: Name of the collection to create.
            debug: Whether to print debug information.
        """
        documents_data = load_pdf(pdf_path)
        if debug:
            print("Loaded Documents Data:", documents_data)

        split_documents_data = split_documents(documents_data)
        if debug:
            print("Split Documents Data:", split_documents_data)

        self.collection = self.vector_db.add_collection(
            collection_name, CustomOpenAIEmbeddings(openai_api_key=aiosettings.openai_api_key.get_secret_value())
        )
        self.vector_db.add_chunks(split_documents_data)

        print(f"Collection '{collection_name}' created and data added.")

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from the database.

        Args:
            collection_name: Name of the collection to delete.
        """
        self.collection = self.vector_db.get_collection(
            collection_name, CustomOpenAIEmbeddings(openai_api_key=aiosettings.openai_api_key.get_secret_value())
        )
        if self.collection:
            self.vector_db.chroma_client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' deleted.")
        else:
            print(f"Collection '{collection_name}' not found.")

    def query_database(self, query_text: str, collection_name: str, debug: bool = False) -> str:
        """Query the database for relevant documents.

        Args:
            query_text: Query text.
            collection_name: Name of the collection to query.
            debug: Whether to print debug information.

        Returns:
            Response from the RAG model.
        """
        collection = self.get_collection(
            collection_name, CustomOpenAIEmbeddings(openai_api_key=aiosettings.openai_api_key.get_secret_value())
        )
        query_context, prompt = query_database(collection, query_text)

        if debug:
            print("Query Context:", query_context)
            print("Prompt:", prompt)

        rag_response = self.model.invoke(query_context, prompt)
        print("RAG Response:", rag_response)

        return rag_response
