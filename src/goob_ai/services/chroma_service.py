# LINK: https://github.com/mlsmall/RAG-Application-with-LangChain
# SOURCE: https://www.linkedin.com/pulse/building-retrieval-augmented-generation-rag-app-langchain-tiwari-stpfc/
# NOTE: This might be the one, take inspiration from the others
from __future__ import annotations

import argparse
import os
import shutil

from dataclasses import dataclass
from typing import List

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from loguru import logger as LOGGER

from goob_ai.aio_settings import aiosettings


HERE = os.path.dirname(__file__)

DATA_PATH = os.path.join(HERE, "...", "data", "chroma", "documents")
CHROMA_PATH = os.path.join(HERE, "...", "data", "chroma", "vectorstorage")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


# Function to perform the query and get the response
def get_response(query_text: str) -> str:
    """Perform the query and get the response.

    Args:
        query_text (str): The query text to search in the database.

    Returns:
        str: The response text based on the query.
    """
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        return "Unable to find matching results."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    return f"Response: {response_text}\nSources: {sources}"


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

    def __init__(self, openai_api_key: str = aiosettings.openai_api_key) -> None:
        """Initialize the CustomOpenAIEmbeddings class.

        Args:
            openai_api_key (str): The API key for accessing OpenAI services.
        """
        super().__init__(openai_api_key=openai_api_key)

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


def generate_data_store() -> None:
    """Generate and store document embeddings in a Chroma vector store.

    This function performs the following steps:
    1. Loads documents from the specified data path.
    2. Splits the loaded documents into smaller chunks.
    3. Saves the chunks into a Chroma vector store for efficient retrieval.
    """
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents() -> List[Document]:
    """Load documents from the specified data path.

    This function loads documents from the specified data path and returns them
    as a list of Document objects.

    Returns:
        List[Document]: The list of loaded documents.
    """
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, filename)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    return documents


def split_text(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks.

    This function takes a list of documents and splits each document into smaller chunks
    using the RecursiveCharacterTextSplitter. The chunks are then returned as a list of
    Document objects.

    Args:
        documents (List[Document]): The list of documents to be split into chunks.

    Returns:
        List[Document]: The list of document chunks.
    """
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks: List[Document] = text_splitter.split_documents(documents)
    LOGGER.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def save_to_chroma(chunks: list[Document]) -> None:
    """Save document chunks to a Chroma vector store.

    This function performs the following steps:
    1. Initializes the embeddings using the OpenAI API key.
    2. Creates a new Chroma database from the document chunks.
    3. Persists the database to the specified directory.

    Args:
        chunks (list of Document): The list of document chunks to be saved.
    """
    # Clear out the database first.
    # if os.path.exists(CHROMA_PATH):
    #     shutil.rmtree(CHROMA_PATH)

    embeddings = CustomOpenAIEmbeddings(openai_api_key=aiosettings.openai_api_key)
    LOGGER.info(embeddings)
    # Create a new DB from the documents.
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    LOGGER.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
