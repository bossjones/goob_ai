# NOTE: https://github.com/apify/actor-vector-database-integrations/blob/master/code/src/vector_stores/chroma.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from loguru import logger as LOGGER
from sqlalchemy import delete, text, update
from sqlalchemy.sql.expression import literal

from goob_ai.gen_ai.vectorstore.base import VectorDbBase
from goob_ai.models.vectorstores.pgvector_input_model import PgvectorIntegration


class PGVectorDatabase(PGVector, VectorDbBase):
    """PGVector database implementation for vector storage and retrieval."""

    def __init__(self, actor_input: PgvectorIntegration, embeddings: Embeddings) -> None:
        """Initialize the PGVectorDatabase.

        Args:
            actor_input: PgvectorIntegration object containing database configuration.
            embeddings: Embeddings object for generating vector representations.
        """
        super().__init__(
            embeddings=embeddings,
            collection_name=actor_input.postgresCollectionName,
            connection=actor_input.postgresSqlConnectionStr,
            use_jsonb=True,
        )
        self._dummy_vector: list[float] = []

    @property
    def dummy_vector(self) -> list[float]:
        """Get a dummy vector for the current embeddings.

        Returns:
            A dummy vector generated by the embeddings object.
        """
        if not self._dummy_vector and self.embeddings:
            self._dummy_vector = self.embeddings.embed_query("dummy")
        return self._dummy_vector

    async def is_connected(self) -> bool:
        """Check if the database connection is established.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError

    def get(self, id_: str) -> Any:
        """Get a document by ID from the database.

        Used only for testing purposes.

        Args:
            id_: The ID of the document to retrieve.

        Returns:
            The retrieved document.

        Raises:
            ValueError: If the collection is not found.
        """
        with self._make_sync_session() as session:
            if not (collection := self.get_collection(session)):
                raise ValueError("Collection not found")

            return (
                session.query(self.EmbeddingStore)
                .where(self.EmbeddingStore.collection_id == collection.uuid)
                .where(self.EmbeddingStore.id == id_)
                .first()
            )

    def get_all_ids(self) -> list[str]:
        """Get all document IDs from the database.

        Used only for testing purposes.

        Returns:
            A list of all document IDs.

        Raises:
            ValueError: If the collection is not found.
        """
        with self._make_sync_session() as session:
            if not (collection := self.get_collection(session)):
                raise ValueError("Collection not found")

            ids = (
                session.query(self.EmbeddingStore.id).filter(self.EmbeddingStore.collection_id == collection.uuid).all()
            )
            return [r[0] for r in ids]

    def get_by_item_id(self, item_id: str) -> list[Document]:
        """Get documents by item ID.

        Args:
            item_id: The item ID to search for.

        Returns:
            A list of documents matching the item ID.

        Raises:
            ValueError: If the collection is not found.
        """
        with self._make_sync_session() as session:
            if not (collection := self.get_collection(session)):
                raise ValueError("Collection not found")

            results = (
                session.query(self.EmbeddingStore)
                .where(self.EmbeddingStore.collection_id == collection.uuid)
                .where(text("(cmetadata ->> 'item_id') = :value").bindparams(value=item_id))
                .all()
            )

        return [Document(page_content="", metadata=r.cmetadata | {"chunk_id": r.id}) for r in results]

    def update_last_seen_at(self, ids: list[str], last_seen_at: Optional[int] = None) -> None:
        """Update the last_seen_at field in the database for the specified IDs.

        Args:
            ids: A list of document IDs to update.
            last_seen_at: The timestamp to set for last_seen_at. If not provided, the current timestamp is used.

        Raises:
            ValueError: If the collection is not found.
        """
        last_seen_at = last_seen_at or int(datetime.now(timezone.utc).timestamp())

        with self._make_sync_session() as session:
            if not (collection := self.get_collection(session)):
                raise ValueError("Collection not found")

            stmt = (
                update(self.EmbeddingStore)
                .where(self.EmbeddingStore.collection_id == literal(str(collection.uuid)))
                .where(self.EmbeddingStore.id.in_(ids))
                .values(cmetadata=text(f"cmetadata || jsonb_build_object('last_seen_at', {last_seen_at})"))
            )
            session.execute(stmt)
            session.commit()

    def delete_expired(self, expired_ts: int) -> None:
        """Delete expired documents from the index.

        Args:
            expired_ts: The expiration timestamp. Documents with last_seen_at older than this timestamp will be deleted.

        Raises:
            ValueError: If the collection is not found.
        """
        with self._make_sync_session() as session:
            if not (collection := self.get_collection(session)):
                raise ValueError("Collection not found")

            stmt = (
                delete(self.EmbeddingStore)
                .where(self.EmbeddingStore.collection_id == literal(str(collection.uuid)))
                .where(text("(cmetadata ->> 'last_seen_at')::int < :value").bindparams(value=expired_ts))
            )
            session.execute(stmt)
            session.commit()

    def delete_all(self) -> None:
        """Delete all documents from the database.

        Used only for testing purposes.
        """
        if ids := self.get_all_ids():
            self.delete(ids=ids, collection_only=True)

    def search_by_vector(self, vector: list[float], k: int = 10_000, filter_: Optional[dict] = None) -> list[Document]:
        """Search for documents by vector similarity.

        Args:
            vector: The query vector to search for.
            k: The maximum number of results to return. Default is 10,000.
            filter_: Optional filter criteria to apply to the search.

        Returns:
            A list of documents matching the search criteria.
        """
        return self.similarity_search_by_vector(vector, k=k, filter=filter_)
