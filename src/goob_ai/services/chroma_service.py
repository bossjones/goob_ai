"""goob_ai.services.chroma_service"""

# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportCallInDefaultInitializer=false
# pylint: disable=no-name-in-module

# pylint: disable=no-member
# LINK: https://github.com/mlsmall/RAG-Application-with-LangChain
# SOURCE: https://www.linkedin.com/pulse/building-retrieval-augmented-generation-rag-app-langchain-tiwari-stpfc/
# NOTE: This might be the one, take inspiration from the others
from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional, Set, Union

import bs4
import chromadb
import pysnooper
import uritools

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_chroma import Chroma as ChromaVectorStore
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger as LOGGER
from tqdm import tqdm

from goob_ai.aio_settings import aiosettings
from goob_ai.utils import file_functions


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


def get_suffix(filename: str) -> str:
    """Get the file extension from the given filename.

    Args:
        filename: The name of the file.

    Returns:
        The file extension in lowercase without the leading period.
    """
    ext = get_file_extension(filename)
    ext_without_period = remove_leading_period(ext)
    LOGGER.debug(f"ext: {ext}, ext_without_period: {ext_without_period}")
    return ext


def get_file_extension(filename: str) -> str:
    """Get the file extension from the given filename.

    Args:
        filename: The name of the file.

    Returns:
        The file extension in lowercase.
    """
    return pathlib.Path(filename).suffix.lower()


def remove_leading_period(ext: str) -> str:
    """Remove the leading period from the file extension.

    Args:
        ext: The file extension.

    Returns:
        The file extension without the leading period.
    """
    return ext.replace(".", "")


def is_pdf(filename: str) -> bool:
    """Check if the given filename has a PDF extension.

    Args:
        filename: The name of the file.

    Returns:
        True if the file has a PDF extension, False otherwise.
    """
    suffix = get_suffix(filename)
    res = suffix in file_functions.PDF_EXTENSIONS
    LOGGER.debug(f"res: {res}")
    return res


def is_txt(filename: str) -> bool:
    """Check if the given filename has a text extension.

    Args:
        filename: The name of the file.

    Returns:
        True if the file has a text extension, False otherwise.
    """
    suffix = get_suffix(filename)
    res = suffix in file_functions.TXT_EXTENSIONS
    LOGGER.debug(f"res: {res}")
    return res


def is_valid_uri(uri: str) -> bool:
    """
    Check if the given URI is valid.

    Args:
        uri (str): The URI to check.

    Returns:
        bool: True if the URI is valid, False otherwise.
    """
    parts = uritools.urisplit(uri)
    return parts.isuri()


def is_github_io_url(filename: str) -> bool:
    """
    Check if the given filename is a valid GitHub Pages URL.

    Args:
        filename (str): The filename to check.

    Returns:
        bool: True if the filename is a valid GitHub Pages URL, False otherwise.
    """
    if re.match(WEBBASE_LOADER_PATTERN, filename) and is_valid_uri(filename):
        LOGGER.debug("selected filetype github.io url, using WebBaseLoader(filename)")
        return True
    return False


def get_rag_loader(filename: str) -> TextLoader | PyMuPDFLoader | WebBaseLoader | None:
    """Get the appropriate loader for the given filename.

    Args:
        filename: The name of the file.

    Returns:
        The loader for the given file type, or None if the file type is not supported.
    """
    if is_github_io_url(f"{filename}"):
        return WebBaseLoader(
            web_paths=(f"{filename}",),
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
        )
    elif is_txt(filename):
        LOGGER.debug("selected filetype txt, using TextLoader(filename)")
        return TextLoader(filename)
    elif is_pdf(filename):
        LOGGER.debug("selected filetype pdf, using PyMuPDFLoader(filename)")
        return PyMuPDFLoader(filename, extract_images=True)
    else:
        LOGGER.debug(f"selected filetype UNKNOWN, using None. uri: {filename}")
        return None


def get_rag_splitter(filename: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> CharacterTextSplitter | None:
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
    LOGGER.debug(f"get_rag_splitter(filename={filename}, chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")

    if is_github_io_url(f"{filename}"):
        LOGGER.debug(
            f"selected filetype github.io url, usingRecursiveCharacterTextSplitter(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
        )
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif is_txt(filename):
        LOGGER.debug(f"selected filetype txt, using CharacterTextSplitter(chunk_size={chunk_size}, chunk_overlap=0)")
        return CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    else:
        LOGGER.debug(f"selected filetype UNKNOWN, using None. uri: {filename}")
        return None


def get_rag_embedding_function(
    filename: str, disallowed_special: Union[Literal["all"], set[str], Sequence[str], None] = None
) -> SentenceTransformerEmbeddings | OpenAIEmbeddings | None:
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

    if is_github_io_url(f"{filename}"):
        LOGGER.debug(
            f"selected filetype github.io url, using OpenAIEmbeddings(disallowed_special={disallowed_special})"
        )
        return OpenAIEmbeddings(disallowed_special=disallowed_special)
    elif is_txt(filename):
        LOGGER.debug(
            f'selected filetype txt, using SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", disallowed_special={disallowed_special})'
        )
        return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    elif is_pdf(filename):
        LOGGER.debug(f"selected filetype pdf, using OpenAIEmbeddings(disallowed_special={disallowed_special})")
        return OpenAIEmbeddings(disallowed_special=disallowed_special)
    else:
        LOGGER.debug(f"selected filetype UNKNOWN, using None. uri: {filename}")
        return None


def get_client(host: str = aiosettings.chroma_host, port: int = aiosettings.chroma_port) -> chromadb.ClientAPI:
    """Get the ChromaDB client.

    Returns:
        The ChromaDB client.
    """
    return chromadb.HttpClient(
        host=host,
        port=port,
        settings=ChromaSettings(allow_reset=True, is_persistent=True),
    )


def search_db(db: Chroma, query_text: str, k: int = 3) -> list[tuple[Document, float]] | None:
    """Search the Chroma database for relevant documents.

    Args:
        db (Chroma): The Chroma database to search.
        query_text (str): The query text to search for.
        k (int): Number of nearest neighbours to return.

    Returns:
        list[tuple[Document, float]] | None: The list of relevant documents and their scores,
        or None if no relevant documents are found.
    """
    results = db.similarity_search_with_relevance_scores(query_text, k=k)
    LOGGER.debug(f"search_db results: {results}")
    if len(results) == 0 or results[0][1] < 0.7:
        return None
    return results


def generate_context_text(results: list[tuple[Document, float]]) -> str:
    """Generate the context text from the search results.

    Args:
        results (list[tuple[Document, float]]): The list of relevant documents and their scores.

    Returns:
        str: The generated context text.
    """
    return "\n\n---\n\n".join([doc.page_content for doc, _score in results])


def generate_prompt(context_text: str, query_text: str) -> str:
    """Generate the prompt for the model.

    Args:
        context_text (str): The context text generated from the search results.
        query_text (str): The query text to search for.

    Returns:
        str: The generated prompt.
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt_template.format(context=context_text, question=query_text)


def get_sources(results: list[tuple[Document, float]]) -> list[str | None]:
    """Get the sources from the search results.

    Args:
        results (list[tuple[Document, float]]): The list of relevant documents and their scores.

    Returns:
        list[str | None]: The list of sources.
    """
    return [doc.metadata.get("source", None) for doc, _score in results]


def get_response(
    query_text: str,
    persist_directory: str = CHROMA_PATH,
    embedding_function: Any = OpenAIEmbeddings(),
    model: Any = ChatOpenAI(),
    k: int = 3,
    collection_name: str = "",
    **kwargs: Any,
) -> str:
    """Perform the query and get the response.

    Args:
        query_text (str): The query text to search in the database.
        persist_directory (str): The directory to persist the Chroma database.
        embedding_function (Any): The embedding function to use.
        **kwargs: Additional keyword arguments to override default values.

    Returns:
        str: The response text based on the query.
    """
    db = get_chroma_db(persist_directory, embedding_function, collection_name=collection_name)

    # Search the DB
    results = search_db(db, query_text, k=k)
    if not results:
        return "Unable to find matching results."

    context_text = generate_context_text(results)
    prompt = generate_prompt(context_text, query_text)

    response_text = model.predict(prompt)

    sources = get_sources(results)
    return f"Response: {response_text}\nSources: {sources}"


def get_chroma_db(
    persist_directory: str = CHROMA_PATH,
    embedding_function: Any = OpenAIEmbeddings(),
    **kwargs: Any,
) -> Chroma:
    """Get the Chroma database.

    Args:
        persist_directory (str): The directory to persist the Chroma database.
        embedding_function (Any): The embedding function to use.
        **kwargs: Additional keyword arguments to override default values.

    Returns:
        Chroma: The Chroma database.
    """
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_function, **kwargs)


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

    def __init__(
        self,
        openai_api_key: str = aiosettings.openai_api_key.get_secret_value(),
        disallowed_special: Union[Literal["all"], set[str], Sequence[str], None] = None,
    ) -> None:
        """Initialize the CustomOpenAIEmbeddings class.

        Args:
            openai_api_key (str): The API key for accessing OpenAI services.
        """
        super().__init__(openai_api_key=openai_api_key, disallowed_special=disallowed_special)

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


@pysnooper.snoop()
def generate_data_store(
    collection_name: str = "", embedding_function: Any = OpenAIEmbeddings()
) -> VectorStoreRetriever:
    """Generate and store document embeddings in a Chroma vector store.

    This function performs the following steps:
    1. Loads documents from the specified data path.
    2. Splits the loaded documents into smaller chunks.
    3. Saves the chunks into a Chroma vector store for efficient retrieval.
    """
    documents = load_documents()
    chunks = split_text(documents)
    retriever: VectorStoreRetriever = save_to_chroma(chunks, collection_name=collection_name)
    return retriever


def generate_and_query_data_store(
    collection_name: str = "", embedding_function: Any = OpenAIEmbeddings()
) -> VectorStoreRetriever:
    retriever: VectorStoreRetriever = generate_data_store(
        collection_name=collection_name, embedding_function=embedding_function
    )
    return retriever


# @pysnooper.snoop()
def load_documents() -> list[Document]:
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
        # loader = PyPDFLoader(f"{filename}", extract_images=True)
        loader = PyMuPDFLoader(f"{filename}", extract_images=True)
        LOGGER.info(f"Loader: {loader}")
        documents.extend(loader.load())
    return documents


# @pysnooper.snoop()
def split_text(
    documents: list[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 100,
    length_function: Callable[[Document], int] = len,
    add_start_index: bool = True,
) -> list[Document]:
    """Split documents into smaller chunks.

    This function takes a list of documents and splits each document into smaller chunks
    using the RecursiveCharacterTextSplitter. The chunks are then returned as a list of
    Document objects.

    Args:
        documents (List[Document]): The list of documents to be split into chunks.

    Returns:
        List[Document]: The list of document chunks.
    """
    LOGGER.debug(
        f"Split text with chunk size: {chunk_size}, chunk overlap: {chunk_overlap}, length function: {length_function}, add start index: {add_start_index}"
    )

    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=add_start_index,
    )
    # aka chunks = all_splits
    chunks: list[Document] = text_splitter.split_documents(documents)
    LOGGER.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


# @pysnooper.snoop()
def save_to_chroma(
    chunks: list[Document],
    disallowed_special: Union[Literal["all"], set[str], Sequence[str], None] = (),
    use_custom_openai_embeddings: bool = False,
    collection_name: str = "",
) -> VectorStoreRetriever:
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

    # default embeddings
    LOGGER.error("default embeddings OpenAIEmbeddings")
    embeddings = OpenAIEmbeddings(
        openai_api_key=aiosettings.openai_api_key.get_secret_value(), disallowed_special=disallowed_special
    )

    # if flag set to use custom embeddings, override
    if use_custom_openai_embeddings:
        LOGGER.error("Using CustomOpenAIEmbeddings")
        embeddings = CustomOpenAIEmbeddings(
            openai_api_key=aiosettings.openai_api_key.get_secret_value(), disallowed_special=disallowed_special
        )

    LOGGER.info(embeddings)

    # Add to vectorDB
    # from_documents = Create a Chroma vectorstore from a list of documents.
    vectorstore: ChromaVectorStore = ChromaVectorStore.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH, collection_name=collection_name
    )
    db = vectorstore

    LOGGER.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

    retriever: VectorStoreRetriever = vectorstore.as_retriever()

    return retriever


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
    def get_or_create_collection(collection_name: str, embedding_function: Any) -> chromadb.Collection:
        """
        Get or create a collection in ChromaDB.

        Args:
            collection_name: Name of the collection.
            embedding_function: Embedding function to use.

        Returns:
            The created collection.
        """
        collection = ChromaService.client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_function
        )
        LOGGER.debug(f"Collection: {collection}")
        return collection

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
    def load_documents() -> list[Document]:
        """
        Load documents from the specified data path.

        Returns:
            List[Document]: The list of loaded documents.
        """
        return load_documents()

    @staticmethod
    def split_text(documents: list[Document]) -> list[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents (List[Document]): The list of documents to be split into chunks.

        Returns:
            List[Document]: The list of document chunks.
        """
        return split_text(documents)

    @staticmethod
    def get_vector_store_from_client(
        collection_name: str | None = "",
        embedding_function: Any | None = None,
        client: chromadb.ClientAPI | None = None,
    ) -> ChromaVectorStore:
        """
        Get a Chroma vector store from the ChromaDB client.

        Args:
            collection_name (str): The name of the collection to retrieve.
            embedding_function (Any, optional): The embedding function to use. Defaults to None.

        Returns:
            ChromaVectorStore: The Chroma vector store.
        """
        # client = ChromaService.get_client()
        collection = ChromaService.get_or_create_collection(collection_name, embedding_function)
        return ChromaVectorStore(client=client, collection_name=collection_name, embedding_function=embedding_function)

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

        # Log the input parameters for debugging purposes
        LOGGER.debug(f"path_to_document = {path_to_document}")
        LOGGER.debug(f"collection_name = {collection_name}")
        LOGGER.debug(f"embedding_function = {embedding_function}")

        # Get the Chroma client
        client = ChromaService.get_client()
        # FIXME: We need to make embedding_function optional
        # Add or retrieve the collection with the specified name
        collection: chromadb.Collection = ChromaService.add_collection(collection_name)

        # Load the document using the appropriate loader based on the file type
        loader: TextLoader | PyMuPDFLoader | WebBaseLoader | None = get_rag_loader(path_to_document)
        # Load the documents using the selected loader
        documents: list[Document] = loader.load()

        # If the file type is txt, split the documents into chunks
        text_splitter = get_rag_splitter(path_to_document)
        if text_splitter:
            # Split the documents into chunks using the text splitter
            docs: list[Document] = text_splitter.split_documents(documents)
        else:
            # If no text splitter is available, use the original documents
            docs: list[Document] = documents  # type: ignore

        if embedding_function:
            # If an embedding function is provided, use it
            embedding_function = embedding_function
        else:
            # If no embedding function is provided, create an open-source embedding function based on the file type
            embedding_function = get_rag_embedding_function(path_to_document)

        # Load the document chunks into Chroma
        db: ChromaVectorStore = Chroma.from_documents(
            docs, embedding=embedding_function, collection_name=collection_name, client=client
        )
        # Return the Chroma database
        return db

    @staticmethod
    def save_to_chroma(chunks: list[Document]) -> None:
        """
        Save document chunks to a Chroma vector store.

        Args:
            chunks (list[Document]): The list of document chunks to be saved.
        """
        save_to_chroma(chunks)

    # https://github.com/langchain-ai/langchain/blob/master/cookbook/img-to_img-search_CLIP_ChromaDB.ipynb
    @staticmethod
    def embed_images(chroma_client: chromadb.ClientAPI | None = None, uris: list[str] = [], metadatas: list[dict] = []):
        """
        Function to add images to Chroma client with progress bar.

        Args:
            chroma_client: The Chroma client object.
            uris (List[str]): List of image file paths.
            metadatas (List[dict]): List of metadata dictionaries.
        """
        if chroma_client is None:
            chroma_client = ChromaService.get_client()

        LOGGER.debug(f"chroma_client: {chroma_client}")
        LOGGER.debug(f"uris: {uris}")

        # Iterate through the uris with a progress bar
        success_count = 0
        for i in tqdm(range(len(uris)), desc="Adding images"):
            uri = uris[i]
            metadata = metadatas[i]

            try:
                chroma_client.add_images(uris=[uri], metadatas=[metadata])
            except Exception as e:
                LOGGER.error(f"Failed to add image {uri} with metadata {metadata}. Error: {e}")
            else:
                success_count += 1
                # print(f"Successfully added image {uri} with metadata {metadata}")

        return success_count


if __name__ == "__main__":
    main()
