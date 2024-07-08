"""goob_ai.services.chroma_service"""

# LINK: https://github.com/mlsmall/RAG-Application-with-LangChain
# SOURCE: https://www.linkedin.com/pulse/building-retrieval-augmented-generation-rag-app-langchain-tiwari-stpfc/
# NOTE: This might be the one, take inspiration from the others
from __future__ import annotations

import argparse
import os
import pathlib
import shutil

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import chromadb

from chromadb.config import Settings as ChromaSettings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma as ChromaVectorStore
from langchain_text_splitters import CharacterTextSplitter
from loguru import logger as LOGGER

from goob_ai.aio_settings import aiosettings
from goob_ai.utils import file_functions
import re
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
import uritools


HERE = os.path.dirname(__file__)

DATA_PATH = os.path.join(HERE, "..", "data", "chroma", "documents")
CHROMA_PATH = os.path.join(HERE, "..", "data", "chroma", "vectorstorage")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Define the regex pattern to match a valid URL containing "github.io"
WEBBASE_LOADER_PATTERN = r"^https?://[a-zA-Z0-9.-]+\.github\.io(/.*)?$"


def get_rag_loader(filename: str) -> TextLoader | PyMuPDFLoader | WebBaseLoader | None:
    """
    Get the appropriate loader for the given filename.

    This function determines the type of the given filename and returns the
    appropriate loader for it. It supports loading from text files, PDF files,
    and URLs matching the pattern for GitHub Pages.

    Args:
        filename (str): The name of the file to load.

    Returns:
        TextLoader | PyMuPDFLoader | WebBaseLoader | None: The loader for the given file,
        or None if the file type is not supported.
    """
def get_rag_embedding_function(filename: str) -> SentenceTransformerEmbeddings | OpenAIEmbeddings | None:
    """
    Get the appropriate embedding function for the given filename.

    This function determines the type of the given filename and returns the
    appropriate embedding function for it. It supports embedding text files,
    PDF files, and URLs matching the pattern for GitHub Pages.

    Args:
        filename (str): The name of the file to embed.

    Returns:
        SentenceTransformerEmbeddings | OpenAIEmbeddings | None: The embedding function for the given file,
        or None if the file type is not supported.
    """
        # verfiy it is a uri as well
        parts = uritools.urisplit(f"{filename}")
        assert parts.isuri()
        LOGGER.debug("selected filetype github.io url, using WebBaseLoader(filename)")
        return WebBaseLoader(
            web_paths=(f"{filename}",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
    elif pathlib.Path(f"{filename}").suffix.lower() in file_functions.TXT_EXTENSIONS:
        LOGGER.debug("selected filetype txt, using TextLoader(filename)")
        return TextLoader(filename)
    elif pathlib.Path(f"{filename}").suffix.lower() in file_functions.PDF_EXTENSIONS:
        LOGGER.debug("selected filetype pdf, using PyMuPDFLoader(filename)")
        return PyMuPDFLoader(filename)
    else:
        LOGGER.debug("selected filetype UNKNOWN, using None")
        return None


def get_rag_splitter(filename: str) -> CharacterTextSplitter | None:
    """
    Get the appropriate text splitter for the given filename.

    This function determines the type of the given filename and returns the
    appropriate text splitter for it. It supports splitting text files and
    URLs matching the pattern for GitHub Pages.

    Args:
        filename (str): The name of the file to split.

    Returns:
        CharacterTextSplitter | None: The text splitter for the given file,
        or None if the file type is not supported.
    """
    if re.match(WEBBASE_LOADER_PATTERN, f"{filename}"):
        LOGGER.debug("selected filetype github.io url, usingRecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)")
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    elif pathlib.Path(f"{filename}").suffix.lower() in file_functions.TXT_EXTENSIONS:
        LOGGER.debug("selected filetype txt, using CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)")
        return CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    else:
        LOGGER.debug("selected filetype UNKNOWN, using None")
        return None


def get_rag_embedding_function(filename: str) -> SentenceTransformerEmbeddings | OpenAIEmbeddings | None:
    if re.match(WEBBASE_LOADER_PATTERN, f"{filename}"):
        LOGGER.debug("selected filetype github.io url, using OpenAIEmbeddings()")
        return OpenAIEmbeddings()
    elif pathlib.Path(f"{filename}").suffix.lower() in file_functions.TXT_EXTENSIONS:
        LOGGER.debug("selected filetype txt, using SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')")
        return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    elif pathlib.Path(f"{filename}").suffix.lower() in file_functions.PDF_EXTENSIONS:
        LOGGER.debug("selected filetype pdf, using OpenAIEmbeddings()")
        return OpenAIEmbeddings()
    else:
        LOGGER.debug(f"selected filetype UNKNOWN, using None")
        return None


def get_client() -> chromadb.ClientAPI:
    """_summary_

    Args:
        query_text (str): _description_

    Returns:
        str: _description_
    """
    return chromadb.HttpClient(
        host=aiosettings.chroma_host, port=aiosettings.chroma_port, settings=ChromaSettings(allow_reset=True)
    )


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

    d = file_functions.tree(DATA_PATH)
    result = file_functions.filter_pdfs(d)

    for filename in result:
        LOGGER.info(f"Loading document: {filename}")
        loader = PyPDFLoader(f"{filename}")
        LOGGER.info(f"Loader: {loader}")
        documents.extend(loader.load())
    # for filename in os.listdir(DATA_PATH):
    #     LOGGER.info(f"Loading document: {filename}")
    #     if filename.endswith(".pdf"):
    #         pdf_path = os.path.join(DATA_PATH, filename)
    #         LOGGER.info(f"Loading PDF: {pdf_path}")
    #         loader = PyPDFLoader(pdf_path)
    #         documents.extend(loader.load())
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
    db = ChromaVectorStore.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    LOGGER.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


class ChromaService:
    """
    Service class for interacting with ChromaDB.

    This class provides static methods to interact with ChromaDB, including
    adding collections, listing collections, and retrieving collections.
    """

    client: chromadb.ClientAPI | None = get_client()
    collection: chromadb.Collection | None = None

    def __init__(self):
        # self.name = "ChromaService"
        # self.client = get_client()
        pass

    @staticmethod
    def add_collection(collection_name: str, embedding_function: Any | None = None) -> chromadb.Collection:
        """
        Add a collection to ChromaDB.

        Args:
            collection_name (str): The name of the collection to add.
            embedding_function (Any): The embedding function to use.

        Returns:
            chromadb.Collection: The created or retrieved collection.
        """
        return (
            ChromaService.client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)
            if embedding_function
            else ChromaService.client.get_or_create_collection(name=collection_name)
        )

    @staticmethod
    def get_list_collections() -> Sequence[chromadb.Collection]:
        """
        List all collections in ChromaDB.

        Returns:
            Sequence[chromadb.Collection]: A sequence of all collections.
        """
        return ChromaService.client.list_collections()

    @staticmethod
    def get_collection(collection_name: str, embedding_function: Any) -> chromadb.Collection | None:
        """
        Retrieve a collection from ChromaDB.

        Args:
            collection_name (str): The name of the collection to retrieve.
            embedding_function (Any): The embedding function to use.

        Returns:
            chromadb.Collection | None: The retrieved collection or None if not found.
        """
        return ChromaService.client.get_collection(name=collection_name, embedding_function=embedding_function)

    @staticmethod
    def get_client() -> chromadb.ClientAPI:
        """
        Get the ChromaDB client.

        Returns:
            chromadb.ClientAPI: The ChromaDB client.
        """
        return ChromaService.client

    @staticmethod
    def get_or_create_collection(query_text: str) -> chromadb.ClientAPI:
        """
        Get or create a collection in ChromaDB.

        Args:
            query_text (str): The query text to search in the database.

        Returns:
            chromadb.ClientAPI: The ChromaDB client.
        """
        return ChromaService.client

    @staticmethod
    def get_response(query_text: str) -> str:
        """
        Get a response from ChromaDB based on the query text.

        Args:
            query_text (str): The query text to search in the database.

        Returns:
            str: The response text based on the query.
        """
        return get_response(query_text)

    @staticmethod
    def generate_data_store() -> None:
        """
        Generate and store document embeddings in a Chroma vector store.
        """
        generate_data_store()

    @staticmethod
    def load_documents() -> List[Document]:
        """
        Load documents from the specified data path.

        Returns:
            List[Document]: The list of loaded documents.
        """
        return load_documents()

    @staticmethod
    def split_text(documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents (List[Document]): The list of documents to be split into chunks.

        Returns:
            List[Document]: The list of document chunks.
        """
        return split_text(documents)

    @staticmethod
    def add_to_chroma(
        path_to_document: str = "", collection_name: str = "", embedding_function: Any | None = None
    ) -> ChromaVectorStore:
        # sourcery skip: inline-immediately-returned-variable, use-named-expression
        """
        Add/Save document chunks to a Chroma vector store.

        Args:
            chunks (list[Document]): The list of document chunks to be saved.
        """

        client = ChromaService.client

        # load the document and split it into chunks
        loader: TextLoader | PyMuPDFLoader | None = get_rag_loader(path_to_document)
        documents: List[Document] = loader.load()

        # If filetype is txt, split it into chunks
        text_splitter = get_rag_splitter(path_to_document)
        if text_splitter:
            docs: List[Document] = text_splitter.split_documents(documents)
        else:
            docs: List[Document] = documents  # type: ignore

        if embedding_function:
            embedding_function = embedding_function
        else:
            # create the open-source embedding function
            embedding_function = get_rag_embedding_function(path_to_document)

        # FIXME: We need to make embedding_function optional
        collection: chromadb.Collection = ChromaService.add_collection(collection_name)

        # load it into Chroma
        db = Chroma.from_documents(docs, embedding_function, collection_name=collection_name, client=client)
        return db

    @staticmethod
    def save_to_chroma(chunks: list[Document]) -> None:
        """
        Save document chunks to a Chroma vector store.

        Args:
            chunks (list[Document]): The list of document chunks to be saved.
        """
        save_to_chroma(chunks)


if __name__ == "__main__":
    main()
