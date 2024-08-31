"""goob_ai.services.pgvector_service"""

# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportCallInDefaultInitializer=false
# pylint: disable=no-name-in-module
# pylint: disable=no-member
from __future__ import annotations

import logging
import os
import uuid

from typing import Any, Callable, List, Literal, Optional, Set, Tuple, Union

from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import _get_embedding_collection_store
from langchain_core.documents import Document
from langchain_postgres import PGVector
from loguru import logger as LOGGER
from sqlalchemy import MetaData, Table, create_engine, select, text, update
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


TABLE_COLLECTION = "langchain_pg_collection"
TABLE_DOCS = "langchain_pg_embedding"
HERE = os.path.dirname(__file__)

DATA_PATH = os.path.join(HERE, "..", "data", "chroma", "documents")


EmbeddingStore = _get_embedding_collection_store()[0]


class PgvectorService:
    """Service class for interacting with PGVector."""

    def __init__(self, connection_string: str) -> None:
        """
        Initialize the PgvectorService.

        Args:
            connection_string: The connection string for the database.
        """
        self.embeddings = OpenAIEmbeddings(openai_api_key=aiosettings.openai_api_key.get_secret_value())
        self.cnx = connection_string
        self.collections: list[str] = []
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
            cosine_distance = self.EmbeddingStore.embedding.cosine_distance(query_vector).label("distance")

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

    def create_collection(
        self,
        collection_name: str,
        documents: list[Document],
        video_metadata: dict[str, str],
        pre_delete_collection: bool = False,
    ) -> tuple[uuid.UUID, list[str]]:
        """
        Creates a new collection with the provided name, documents and video metadata.

        Args:
            collection_name: The name of the collection to create.
            documents: The list of documents to add to the collection.
            video_metadata: The metadata associated with the video.
            pre_delete_collection: Set to True to delete the collection if it already exists.

        Returns:
            A tuple containing the UUID of the created collection and a list of document IDs.
        """
        LOGGER.info(f"Deleting collection: {collection_name}")
        with self.engine.connect() as connection:
            collection = PGVector(
                collection_name=collection_name,
                connection_string=self.cnx,
                embedding_function=self.embeddings,
                use_jsonb=True,
                connection=connection,
                pre_delete_collection=pre_delete_collection,
            )

            collection_id = self.get_collection_id_by_name(collection_name)
            doc_ids = collection.add_documents(documents)

            return collection_id, doc_ids

    def get_collection_id_by_name(self, collection_name: str, pre_delete_collection: bool = False) -> Optional[str]:
        """
        Fetch the collection ID for the given name.

        Args:
            collection_name: The name of the collection.
            pre_delete_collection: Set to True to delete the collection if it already exists.

        Returns:
            The UUID of the collection if found, otherwise None.
        """
        LOGGER.info(f"getting collection id for: {collection_name}")
        with self.engine.connect() as connection:
            table = Table(TABLE_COLLECTION, MetaData(), autoload_with=connection)
            query = select(table.c.uuid).where(table.c.name == collection_name)
            result = connection.execute(query).fetchone()

            return result[0] if result else None

    def get_collection_metadata(self, collection_id: str, pre_delete_collection: bool = False) -> Optional[dict]:
        """
        Fetch the collection metadata for the given ID.

        Args:
            collection_id: The UUID of the collection.
            pre_delete_collection: Set to True to delete the collection if it already exists.

        Returns:
            The metadata of the collection if found, otherwise None.
        """
        LOGGER.info(f"getting collection metadata for collection id: {collection_id}")
        with self.engine.connect() as connection:
            table = Table(TABLE_COLLECTION, MetaData(), autoload_with=connection)
            query = select(table.c.cmetadata).where(table.c.uuid == collection_id)
            result = connection.execute(query).fetchone()

        return result[0] if result else None

    def update_collection_metadata(self, collection_id: str, new_metadata: dict) -> Optional[dict]:
        """
        Updates the metadata of the collection.

        Args:
            collection_id: The UUID of the collection.
            new_metadata: The new metadata to update.

        Returns:
            The updated metadata of the collection.
        """
        LOGGER.info(f"updating collection metadata for collection id: {collection_id}")
        with self.engine.connect() as connection:
            table = Table(TABLE_COLLECTION, MetaData(), autoload_with=connection)
            query = update(table).where(table.c.uuid == collection_id).values(cmetadata=new_metadata)

            connection.execute(query)
            connection.commit()

        return self.get_collection_metadata(collection_id)

    def list_collections(self) -> list[str]:
        """
        Returns a list of all collections in the vector store.

        Returns:
            A list of collection names.
        """
        LOGGER.info("listing collections")
        with self.engine.connect() as connection:
            table = Table(TABLE_COLLECTION, MetaData(), autoload_with=connection)
            query = select(table.c["name"])
            results = connection.execute(query).fetchall()

        return results

    def get_by_ids(self, ids: list[str]) -> list[str]:
        """
        Returns all documents with the provided IDs.

        Args:
            ids: The list of document IDs.

        Returns:
            A list of documents.
        """
        LOGGER.info(f"get documents by id: {ids}")
        with self.engine.connect() as connection:
            table = Table(TABLE_DOCS, MetaData(), autoload_with=connection)
            query = select(table).where(table.c.id.in_(ids))
            results = connection.execute(query).fetchall()

        return results

    def get_all_by_collection_id(self, collection_id: str) -> list[str]:
        """
        Returns all documents of the collection.

        Args:
            collection_id: The UUID of the collection.

        Returns:
            A list of documents.
        """
        LOGGER.info(f"get documents by id: {collection_id}")
        with self.engine.connect() as connection:
            table = Table(TABLE_DOCS, MetaData(), autoload_with=connection)
            query = select(table).where(table.c.collection_id == collection_id)
            results = connection.execute(query).fetchall()

        return results

    def drop_tables(self, collection_name: str) -> None:
        """
        Delete a collection based on the collection name.

        Args:
            collection_name: The name of the collection to delete.
        """
        LOGGER.info(f"Deleting collection: {collection_name}")
        with self.engine.connect() as connection:
            store = PGVector(
                collection_name=collection_name,
                connection_string=self.cnx,
                connection=connection,
                embedding_function=self.embeddings,
                use_jsonb=True,
            )
            store.drop_tables()

    def reset(self, collection_name: str) -> None:
        """
        Delete a collection based on the collection name.

        Args:
            collection_name: The name of the collection to delete.
        """
        LOGGER.info(f"Deleting collection: {collection_name}")
        with self.engine.connect() as connection:
            store = PGVector(
                collection_name=collection_name,
                connection_string=self.cnx,
                connection=connection,
                embedding_function=self.embeddings,
                use_jsonb=True,
            )
            store.drop_tables()
            store.create_tables_if_not_exists()
            store.create_collection()

    def get_vector_store(self, collection_name: str, metadatas: list[dict[str, Any]] = []) -> PGVector:
        """
        Get the vector store.

        Returns:
            The vector store.
        """
        LOGGER.info(f"getting vector store for collection: {collection_name}")
        LOGGER.info(f"metadatas: {metadatas}")

        if metadatas:
            return PGVector(
                collection_name=collection_name,
                connection_string=self.cnx,
                collection_metadata=metadatas,
                embedding_function=self.embeddings,
                use_jsonb=True,
            )
        else:
            return PGVector(
                collection_name=collection_name,
                connection_string=self.cnx,
                embedding_function=self.embeddings,
                use_jsonb=True,
            )
