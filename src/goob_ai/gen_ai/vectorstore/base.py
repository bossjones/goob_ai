# NOTE: https://github.com/apify/actor-vector-database-integrations/blob/master/code/src/vector_stores/chroma.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

from loguru import logger as LOGGER


if TYPE_CHECKING:
    from langchain_core.documents import Document


class VectorDbBase(ABC):
    """Base class for vector database implementations."""

    # only for testing purposes (to wait for the index to be updated, e.g. in Pinecone)
    unit_test_wait_for_index = 0

    @abstractmethod
    def get_by_item_id(self, item_id: str) -> list[Document]:
        """Get documents by item_id.

        Args:
            item_id: The ID of the item to retrieve documents for.

        Returns:
            A list of documents associated with the given item_id.
        """

    @abstractmethod
    def update_last_seen_at(self, ids: list[str], last_seen_at: Optional[int] = None) -> None:
        """Update last_seen_at field in the database.

        Args:
            ids: A list of document IDs to update the last_seen_at field for.
            last_seen_at: The timestamp to set for the last_seen_at field. If None, the current timestamp is used.
        """

    @abstractmethod
    def delete_expired(self, expired_ts: int) -> None:
        """Delete documents that are older than the ts_expired timestamp.

        Args:
            expired_ts: The timestamp threshold for deleting expired documents.
        """

    @abstractmethod
    def delete_all(self) -> None:
        """Delete all documents from the database (internal function for testing purposes)."""

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if the database is connected.

        Returns:
            True if the database is connected, False otherwise.
        """

    @abstractmethod
    def search_by_vector(self, vector: list[float], k: int, filter_: Optional[dict] = None) -> list[Document]:
        """Search for documents by vector.

        Args:
            vector: The vector to search for.
            k: The number of documents to return.
            filter_: Optional filter criteria to apply to the search.

        Returns:
            A list of documents that match the search criteria.
        """


# NOTE: https://github.com/apify/actor-vector-database-integrations/blob/877b8b45d600eebd400a01533d29160cad348001/code/src/vector_stores/base.py
