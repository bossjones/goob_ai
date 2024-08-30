# NOTE: https://github.com/apify/actor-vector-database-integrations/blob/master/code/src/vector_stores/chroma.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from loguru import logger as LOGGER
from pinecone.grpc.pinecone import PineconeGRPC as PineconeClient

from goob_ai.gen_ai.vectorstore.base import VectorDbBase


if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    # from ..models import PineconeIntegration
    from goob_ai.models.vectorstores.pinecone_input_model import PineconeIntegration

# Pinecone API attribution tag
PINECONE_SOURCE_TAG = "apify"

# from pinecone import Pinecone as PineconeClient  # type: ignore[import-untyped]


class PineconeDatabase(PineconeVectorStore, VectorDbBase):
    """Pinecone database wrapper for vector storage and retrieval."""

    def __init__(self, actor_input: PineconeIntegration, embeddings: Embeddings) -> None:
        """Initialize the Pinecone database.

        Args:
            actor_input: Pinecone integration settings.
            embeddings: Embeddings object for encoding vectors.
        """
        self.client = PineconeClient(api_key=actor_input.pineconeApiKey, source_tag=PINECONE_SOURCE_TAG)
        self.index = self.client.Index(actor_input.pineconeIndexName)
        super().__init__(index=self.index, embedding=embeddings)
        self._dummy_vector: list[float] = []

    @property
    def dummy_vector(self) -> list[float]:
        """Get a dummy vector for similarity search.

        Returns:
            A dummy vector.
        """
        if not self._dummy_vector and self.embeddings:
            self._dummy_vector = self.embeddings.embed_query("dummy")
        return self._dummy_vector

    async def is_connected(self) -> bool:
        """Check if the database is connected.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError

    def get_by_item_id(self, item_id: str) -> list[Document]:
        """Get documents by item_id.

        Args:
            item_id: The item ID to search for.

        Returns:
            A list of documents matching the item ID.
        """
        results = self.index.query(
            vector=self.dummy_vector, top_k=10_000, filter={"item_id": item_id}, include_metadata=True
        )
        return [Document(page_content="", metadata=d["metadata"] | {"chunk_id": d["id"]}) for d in results["matches"]]

    def update_last_seen_at(self, ids: list[str], last_seen_at: Optional[int] = None) -> None:
        """Update the last_seen_at field for the given IDs.

        Args:
            ids: The list of IDs to update.
            last_seen_at: The timestamp to set. Defaults to the current timestamp.
        """
        last_seen_at = last_seen_at or int(datetime.now(timezone.utc).timestamp())
        for _id in ids:
            self.index.update(id=_id, set_metadata={"last_seen_at": last_seen_at})

    def delete_expired(self, expired_ts: int) -> None:
        """Delete expired documents from the index.

        Args:
            expired_ts: The expiration timestamp.
        """
        res = self.search_by_vector(self.dummy_vector, filter_={"last_seen_at": {"$lt": expired_ts}})
        ids = [d.metadata.get("id") or d.metadata.get("chunk_id", "") for d in res]
        ids = [_id for _id in ids if _id]
        self.delete(ids=ids)

    def delete_all(self) -> None:
        """Delete all documents from the index."""
        if r := list(self.index.list(prefix="")):
            self.delete(ids=r)

    def search_by_vector(self, vector: list[float], k: int = 10_000, filter_: Optional[dict] = None) -> list[Document]:
        """Search documents by vector similarity.

        Args:
            vector: The query vector.
            k: The number of results to return. Defaults to 10_000.
            filter_: Additional filter criteria. Defaults to None.

        Returns:
            A list of similar documents.
        """
        res = self.similarity_search_by_vector_with_score(vector, k=k, filter=filter_)
        return [r for r, _ in res]
