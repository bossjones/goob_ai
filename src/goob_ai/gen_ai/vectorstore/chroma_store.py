# NOTE: https://github.com/apify/actor-vector-database-integrations/blob/master/code/src/vector_stores/chroma.py
from __future__ import annotations

from datetime import UTC, datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

# TODO: Remove chromadb dependency once reimplemented
# import chromadb
# from chromadb.config import Settings as ChromaSettings
# from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger as LOGGER

from goob_ai.aio_settings import aiosettings
from goob_ai.gen_ai.vectorstore.base import VectorDbBase

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from goob_ai.models.vectorstores.chroma_input_model import ChromaIntegration


# TODO: Reimplement ChromaDatabase without langchain_chroma dependency
class ChromaDatabase(VectorDbBase):  # Removed Chroma inheritance
    """Chroma database wrapper for vector storage and retrieval."""

    def __init__(self, actor_input: ChromaIntegration, embeddings: Embeddings) -> None:
        """Initialize the ChromaDatabase.

        Args:
            actor_input: ChromaIntegration object containing configuration settings.
            embeddings: Embeddings object for generating vector representations.
        """
        # TODO: Reimplement initialization without chromadb dependency
        raise NotImplementedError("ChromaDatabase is currently disabled")
        # settings = None
        # if auth := actor_input.chromaServerAuthCredentials:
        #     settings = ChromaSettings(
        #         chroma_client_auth_credentials=auth,
        #         chroma_client_auth_provider=actor_input.chromaClientAuthProvider,
        #     )
        # client = chromadb.HttpClient(
        #     host=actor_input.chromaClientHost,
        #     port=actor_input.chromaClientPort or 8000,
        #     ssl=actor_input.chromaClientSsl or False,
        #     settings=settings,
        # )
        # collection_name = actor_input.chromaCollectionName or "chroma"
        # super().__init__(
        #     client=client,
        #     collection_name=collection_name,
        #     embedding_function=embeddings,
        # )
        # self.client = client
        # self.index = self.client.get_collection(collection_name)
        # self._dummy_vector: list[float] = []

    @property
    def dummy_vector(self) -> list[float]:
        """Get a dummy vector for initialization purposes.

        Returns:
            A dummy vector generated from the embeddings.
        """
        # TODO: Reimplement dummy_vector without chromadb dependency
        raise NotImplementedError("dummy_vector is currently disabled")
        # if not self._dummy_vector and self.embeddings:
        #     self._dummy_vector = self.embeddings.embed_query("dummy")
        # return self._dummy_vector

    async def is_connected(self) -> bool:
        """Check if the database is connected.

        Returns:
            True if the database is connected, False otherwise.
        """
        # TODO: Reimplement is_connected without chromadb dependency
        raise NotImplementedError("is_connected is currently disabled")
        # if self.client.heartbeat() <= 1:
        #     return False
        # return True

    def get_by_item_id(self, item_id: str) -> list[Document]:
        """Get documents by item_id.

        Args:
            item_id: The item_id to retrieve documents for.

        Returns:
            A list of Document objects matching the item_id.
        """
        # TODO: Reimplement get_by_item_id without chromadb dependency
        raise NotImplementedError("get_by_item_id is currently disabled")
        # results = self.index.get(where={"item_id": item_id}, include=["metadatas"])
        # if (ids := results.get("ids")) and (metadata := results.get("metadatas")):
        #     return [
        #         Document(page_content="", metadata={**m, "chunk_id": _id})
        #         for _id, m in zip(ids, metadata, strict=False)
        #     ]
        # return []

    def update_last_seen_at(self, ids: list[str], last_seen_at: int | None = None) -> None:
        """Update last_seen_at field in the database.

        Args:
            ids: List of document IDs to update.
            last_seen_at: Timestamp to set for last_seen_at. Defaults to current timestamp.
        """
        # TODO: Reimplement update_last_seen_at without chromadb dependency
        raise NotImplementedError("update_last_seen_at is currently disabled")
        # last_seen_at = last_seen_at or int(datetime.now(UTC).timestamp())
        # for _id in ids:
        #     self.index.update(ids=_id, metadatas=[{"last_seen_at": last_seen_at}])

    def delete_expired(self, expired_ts: int) -> None:
        """Delete expired objects.

        Args:
            expired_ts: Timestamp threshold for expiration.
        """
        # TODO: Reimplement delete_expired without chromadb dependency
        raise NotImplementedError("delete_expired is currently disabled")
        # self.index.delete(where={"last_seen_at": {"$lt": expired_ts}})  # type: ignore[dict-item]

    def delete_all(self) -> None:
        """Delete all objects in the database."""
        # TODO: Reimplement delete_all without chromadb dependency
        raise NotImplementedError("delete_all is currently disabled")
        # r = self.index.get()
        # if r["ids"]:
        #     self.delete(ids=r["ids"])

    def search_by_vector(self, vector: list[float], k: int = 1_000_000, filter_: dict | None = None) -> list[Document]:
        """Search documents by vector similarity.

        Args:
            vector: The query vector to search for.
            k: The maximum number of results to return. Defaults to 1,000,000.
            filter_: Optional filter criteria for the search. Defaults to None.

        Returns:
            A list of Document objects most similar to the query vector.
        """
        # TODO: Reimplement search_by_vector without chromadb dependency
        raise NotImplementedError("search_by_vector is currently disabled")
        # return self.similarity_search_by_vector(vector, k=k, filter=filter_)
