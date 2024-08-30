"""goob_ai.services.pgvector_service"""

# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportCallInDefaultInitializer=false
# pylint: disable=no-name-in-module
# pylint: disable=no-member
from __future__ import annotations

import logging
import os

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Set, Tuple, Union

from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector, _get_embedding_collection_store
from langchain_core.documents import Document
from loguru import logger as LOGGER
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from goob_ai.aio_settings import aiosettings
from goob_ai.gen_ai.utilities import (
    WEBBASE_LOADER_PATTERN,
    calculate_chunk_ids,
    franchise_metadata,
    generate_document_hashes,
    get_file_extension,
    get_nested_value,
    get_rag_embedding_function,
    get_rag_loader,
    get_rag_splitter,
    get_suffix,
    is_github_io_url,
    is_pdf,
    is_txt,
    is_valid_uri,
    markdown_to_documents,
    remove_leading_period,
    string_to_doc,
    stringify_dict,
)
from goob_ai.utils import file_functions


HERE = os.path.dirname(__file__)

DATA_PATH = os.path.join(HERE, "..", "data", "chroma", "documents")


EmbeddingStore = _get_embedding_collection_store()[0]


class PgvectorService:
    """Service class for interacting with PGVector."""

    def __init__(self, connection_string: str):
        """
        Initialize the PgvectorService.

        Args:
            connection_string: The connection string for the database.
        """
        # load_dotenv()
        self.embeddings = OpenAIEmbeddings(openai_api_key=aiosettings.openai_api_key.get_secret_value())
        self.cnx = connection_string
        self.collections = []
        self.engine = create_engine(self.cnx)
        self.EmbeddingStore = EmbeddingStore

    def get_vector(self, text: str) -> list[float]:
        """
        Get the vector representation of the given text.

        Args:
            text: The text to get the vector representation for.

        Returns:
            The vector representation of the text.
        """
        return self.embeddings.embed_query(text)

    def custom_similarity_search_with_scores(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        """
        Perform a custom similarity search with scores.

        Args:
            query: The query text.
            k: The number of results to return.

        Returns:
            A list of tuples containing the matched documents and their similarity scores.
        """
        query_vector = self.get_vector(query)

        with Session(self.engine) as session:
            # Using cosine similarity for the vector comparison
            cosine_distance = self.EmbeddingStore.embedding.cosine_distance(query_vector).label("distance")

            # Querying the EmbeddingStore table
            results = (
                session.query(
                    self.EmbeddingStore.document,
                    self.EmbeddingStore.custom_id,
                    cosine_distance,
                )
                .order_by(cosine_distance.asc())
                .limit(k)
                .all()
            )
        # Calculate the similarity score by subtracting the cosine distance from 1 (_cosine_relevance_score_fn)
        docs = [(Document(page_content=result[0]), 1 - result[2]) for result in results]

        return docs

    def update_pgvector_collection(self, docs: list[Document], collection_name: str, overwrite: bool = False) -> None:
        """
        Create a new collection from documents.

        Args:
            docs: The list of documents to create the collection from.
            collection_name: The name of the collection.
            overwrite: Set to True to delete the collection if it already exists.
        """
        LOGGER.info(f"Creating new collection: {collection_name}")
        with self.engine.connect() as connection:
            pgvector = PGVector.from_documents(
                embedding=self.embeddings,
                documents=docs,
                collection_name=collection_name,
                connection_string=self.cnx,
                connection=connection,
                pre_delete_collection=overwrite,
            )

    def get_collections(self) -> list[str]:
        """
        Get the list of existing collections.

        Returns:
            A list of collection names.
        """
        with self.engine.connect() as connection:
            try:
                query = text("SELECT * FROM public.langchain_pg_collection")
                result = connection.execute(query)
                collections = [row[0] for row in result]
            except:
                # If the table doesn't exist, return an empty list
                collections = []
        return collections

    def update_collection(self, docs: list[Document], collection_name: str) -> None:
        """
        Update a collection with data from a given list of documents.

        Args:
            docs: The list of documents to update the collection with.
            collection_name: The name of the collection to update.
        """
        LOGGER.info(f"Updating collection: {collection_name}")
        collections = self.get_collections()

        if docs is not None:
            overwrite = collection_name in collections
            self.update_pgvector_collection(docs, collection_name, overwrite)

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection based on the collection name.

        Args:
            collection_name: The name of the collection to delete.
        """
        LOGGER.info(f"Deleting collection: {collection_name}")
        with self.engine.connect() as connection:
            pgvector = PGVector(
                collection_name=collection_name,
                connection_string=self.cnx,
                connection=connection,
                embedding_function=self.embeddings,
            )
            pgvector.delete_collection()
