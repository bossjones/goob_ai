from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Any

from goob_ai.aio_settings import aiosettings
from goob_ai.services.pgvector_service import PgvectorService
from langchain_core.documents import Document
from loguru import logger as LOGGER
from sqlalchemy.orm import Session

import pytest


if TYPE_CHECKING:
    from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock

    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture()
def pgvector_service() -> PgvectorService:
    """Fixture to create a PgvectorService instance.

    Returns:
        PgvectorService: An instance of PgvectorService.
    """
    return PgvectorService(aiosettings.postgres_url)


@pytest.mark.asyncio()
@pytest.mark.services()
async def test_get_vector(pgvector_service: PgvectorService) -> None:
    """Test the get_vector method.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
    """
    text = "Test text"
    vector = pgvector_service.get_vector(text)
    assert isinstance(vector, list)
    assert len(vector) > 0


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
@pytest.mark.asyncio()
@pytest.mark.services()
async def test_custom_similarity_search_with_scores(pgvector_service: PgvectorService, mocker) -> None:
    """Test the custom_similarity_search_with_scores method.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        mocker: The pytest-mock plugin for mocking.
    """
    query = "Test query"
    k = 3

    # Mock the Session and query results
    mock_session = mocker.Mock(spec=Session)
    mock_session.__enter__ = mocker.Mock(return_value=(mocker.Mock(), None))
    mock_session.__exit__ = mocker.Mock(return_value=None)
    mock_query = mock_session.query.return_value
    mock_query.order_by.return_value.limit.return_value.all.return_value = [
        ("Document 1", "id1", 0.1),
        ("Document 2", "id2", 0.2),
        ("Document 3", "id3", 0.3),
    ]

    mocker.patch("goob_ai.services.pgvector_service.Session", return_value=mock_session)
    results = pgvector_service.custom_similarity_search_with_scores(query, k)

    assert len(results) == 3
    assert all(isinstance(doc, Document) and isinstance(score, float) for doc, score in results)


@pytest.mark.asyncio()
@pytest.mark.services()
async def test_update_pgvector_collection(pgvector_service: PgvectorService, mocker) -> None:
    """Test the update_pgvector_collection method.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        mocker: The pytest-mock plugin for mocking.
    """
    docs = [Document(page_content="Test document")]
    collection_name = "test_collection"

    mock_pgvector = mocker.patch("goob_ai.services.pgvector_service.PGVector")
    pgvector_service.update_pgvector_collection(docs, collection_name)
    mock_pgvector.from_documents.assert_called_once()


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
@pytest.mark.asyncio()
@pytest.mark.services()
async def test_get_collections(pgvector_service: PgvectorService, mocker) -> None:
    """Test the get_collections method.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        mocker: The pytest-mock plugin for mocking.
    """
    mock_result = mocker.Mock()
    mock_result.fetchall.return_value = [("collection1",), ("collection2",)]

    mock_text = mocker.patch("goob_ai.services.pgvector_service.text")
    mock_connect = mocker.patch.object(pgvector_service.engine, "connect")
    mock_connect.return_value.__enter__.return_value.execute.return_value = mock_result
    collections = pgvector_service.get_collections()

    assert collections == ["collection1", "collection2"]
    mock_text.assert_called_once_with("SELECT * FROM public.langchain_pg_collection")


@pytest.mark.asyncio()
@pytest.mark.services()
async def test_update_collection(pgvector_service: PgvectorService, mocker) -> None:
    """Test the update_collection method.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        mocker: The pytest-mock plugin for mocking.
    """
    docs = [Document(page_content="Test document")]
    collection_name = "test_collection"

    mocker.patch.object(pgvector_service, "get_collections", return_value=["existing_collection"])
    mock_update = mocker.patch.object(pgvector_service, "update_pgvector_collection")
    pgvector_service.update_collection(docs, collection_name)
    mock_update.assert_called_once_with(docs, collection_name, False)


@pytest.mark.asyncio()
@pytest.mark.services()
async def test_delete_collection(pgvector_service: PgvectorService, mocker) -> None:
    """Test the delete_collection method.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        mocker: The pytest-mock plugin for mocking.
    """
    collection_name = "test_collection"

    mock_pgvector = mocker.patch("goob_ai.services.pgvector_service.PGVector")
    pgvector_service.delete_collection(collection_name)
    mock_pgvector.return_value.delete_collection.assert_called_once()


from goob_ai.services.pgvector_service import PgvectorService
from langchain_core.documents import Document

import pytest


@pytest.mark.vcronly()
@pytest.mark.vcr(
    allow_playback_repeats=True, match_on=["method", "scheme", "port", "path", "query"], ignore_localhost=False
)
def test_integration_update_pgvector_collection(pgvector_service: PgvectorService, vcr: Any) -> None:
    """Test the update_pgvector_collection method with VCR.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        vcr (Any): The VCR fixture.
    """
    docs = [Document(page_content="Test document for VCR")]
    collection_name = "test_vcr_collection"

    pgvector_service.update_pgvector_collection(docs, collection_name)

    assert vcr.play_count == 1


@pytest.mark.vcronly()
@pytest.mark.vcr(
    allow_playback_repeats=True, match_on=["method", "scheme", "port", "path", "query"], ignore_localhost=False
)
def test_integration_get_collections(pgvector_service: PgvectorService, vcr: Any) -> None:
    """Test the get_collections method with VCR.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        vcr (Any): The VCR fixture.
    """
    collections = pgvector_service.get_collections()

    assert isinstance(collections, list)
    assert vcr.play_count == 1


@pytest.mark.vcronly()
@pytest.mark.vcr(
    allow_playback_repeats=True, match_on=["method", "scheme", "port", "path", "query"], ignore_localhost=False
)
def test_integration_update_collection(pgvector_service: PgvectorService, vcr: Any) -> None:
    """Test the update_collection method with VCR.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        vcr (Any): The VCR fixture.
    """
    docs = [Document(page_content="Test document for VCR update")]
    collection_name = "test_vcr_update_collection"

    pgvector_service.update_collection(docs, collection_name)

    assert vcr.play_count == 1


@pytest.mark.vcronly()
@pytest.mark.vcr(
    allow_playback_repeats=True, match_on=["method", "scheme", "port", "path", "query"], ignore_localhost=False
)
def test_integration_delete_collection(pgvector_service: PgvectorService, vcr: Any) -> None:
    """Test the delete_collection method with VCR.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        vcr (Any): The VCR fixture.
    """
    collection_name = "test_vcr_delete_collection"

    # First, create a collection to delete
    docs = [Document(page_content="Test document for VCR delete")]
    pgvector_service.update_collection(docs, collection_name)

    # Now delete the collection
    pgvector_service.delete_collection(collection_name)

    assert vcr.play_count == 2  # One for creation, one for deletion


@pytest.mark.vcronly()
@pytest.mark.vcr(
    allow_playback_repeats=True, match_on=["method", "scheme", "port", "path", "query"], ignore_localhost=False
)
def test_integration_create_collection(pgvector_service: PgvectorService, vcr: Any) -> None:
    """Test the create_collection method with VCR.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        vcr (Any): The VCR fixture.
    """
    collection_name = "test_vcr_create_collection"
    documents = [Document(page_content="Test document for VCR create")]
    video_metadata = {"title": "Test Video", "duration": "10:00"}

    collection_id, doc_ids = pgvector_service.create_collection(collection_name, documents, video_metadata)

    assert isinstance(collection_id, str)
    assert isinstance(doc_ids, list)
    assert len(doc_ids) == 1
    assert vcr.play_count == 1
