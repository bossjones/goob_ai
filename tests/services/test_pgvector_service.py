from __future__ import annotations

import logging
import re
import uuid

from typing import TYPE_CHECKING, Any
from uuid import UUID

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


def valid_uuid(uuid_txt: str) -> bool:
    regex = re.compile(r"^[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?[89ab][a-f0-9]{3}-?[a-f0-9]{12}\Z", re.I)
    match = regex.match(uuid_txt)
    return bool(match)


@pytest.fixture()
def pgvector_service() -> PgvectorService:
    """Fixture to create a PgvectorService instance.

    Returns:
        PgvectorService: An instance of PgvectorService.
    """
    return PgvectorService(aiosettings.postgres_url)


@pytest.mark.asyncio()
@pytest.mark.services()
@pytest.mark.pgvectoronly()
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
@pytest.mark.pgvectoronly()
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
@pytest.mark.pgvectoronly()
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
@pytest.mark.pgvectoronly()
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
@pytest.mark.pgvectoronly()
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
@pytest.mark.pgvectoronly()
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


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
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


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
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


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
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


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
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


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky()
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


@pytest.mark.asyncio()
@pytest.mark.services()
@pytest.mark.pgvectoronly()
async def test_integration_get_collection_id_by_name(pgvector_service: PgvectorService, mocker: MockerFixture) -> None:
    """Test the get_collection_id_by_name method.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        mocker (MockerFixture): The pytest-mock plugin for mocking.
    """
    collection_name = "test_collection"
    expected_collection_id = "test_collection_id"

    store = pgvector_service.get_vector_store(collection_name)
    store.__post_init__()
    engine = pgvector_service.engine
    spy = mocker.spy(engine, "connect")

    collection_id = pgvector_service.get_collection_id_by_name(collection_name)

    # assert collection_id == expected_collection_id

    # assert the uuid we get back is valid
    assert UUID(str(collection_id))
    spy.assert_called_once()


@pytest.mark.asyncio()
@pytest.mark.services()
@pytest.mark.pgvectoronly()
async def test_integration_get_collection_metadata(pgvector_service: PgvectorService, mocker: MockerFixture) -> None:
    """Test the get_collection_metadata method.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        mocker (MockerFixture): The pytest-mock plugin for mocking.
    """
    expected_metadata = {"duration": "10:00"}
    collection_name = "test_integration_get_collection_metadata"
    store = pgvector_service.get_vector_store(collection_name, metadatas=expected_metadata)
    store.__post_init__()

    documents = [Document(page_content="Test document for metadata")]

    real_collection_id, doc_ids = pgvector_service.create_collection(collection_name, documents, expected_metadata)

    collection_id = pgvector_service.get_collection_id_by_name(collection_name)

    engine = pgvector_service.engine
    spy = mocker.spy(engine, "connect")

    metadata = pgvector_service.get_collection_metadata(collection_id)

    assert metadata == expected_metadata
    spy.assert_called_once()


@pytest.mark.asyncio()
@pytest.mark.services()
@pytest.mark.pgvectoronly()
async def test_integration_update_collection_metadata(pgvector_service: PgvectorService, mocker: MockerFixture) -> None:
    """Test the update_collection_metadata method.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        mocker (MockerFixture): The pytest-mock plugin for mocking.
    """
    collection_name = "test_collection_update_metadata"
    collection_id = "test_collection_id"
    new_metadata = {"new_key": "new_value"}
    expected_metadata = {"key": "value", "new_key": "new_value"}

    store = pgvector_service.get_vector_store(collection_name)
    store.__post_init__()
    engine = pgvector_service.engine
    spy_connect = mocker.spy(engine, "connect")

    documents = [Document(page_content="Test document for VCR create")]
    video_metadata = {"title": "Test Video", "duration": "10:00"}

    real_collection_id, doc_ids = pgvector_service.create_collection(collection_name, documents, video_metadata)

    spy_get_metadata = mocker.spy(pgvector_service, "get_collection_metadata")
    # spy_connect = mocker.spy(pgvector_service.engine, "connect")

    updated_metadata = pgvector_service.update_collection_metadata(collection_id, new_metadata)

    assert updated_metadata == expected_metadata
    spy_get_metadata.assert_called_once_with(collection_id)
    spy_connect.assert_called_once()


@pytest.mark.asyncio()
@pytest.mark.services()
@pytest.mark.pgvectoronly()
async def test_integration_list_collections(pgvector_service: PgvectorService, mocker: MockerFixture) -> None:
    """Test the list_collections method.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        mocker (MockerFixture): The pytest-mock plugin for mocking.
    """
    expected_collections = [("collection1",), ("collection2",)]

    spy = mocker.spy(pgvector_service.engine, "connect")

    collections = pgvector_service.list_collections()

    assert collections == expected_collections
    spy.assert_called_once()


@pytest.mark.asyncio()
@pytest.mark.services()
@pytest.mark.pgvectoronly()
async def test_integration_get_by_ids(pgvector_service: PgvectorService, mocker: MockerFixture) -> None:
    """Test the get_by_ids method.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        mocker (MockerFixture): The pytest-mock plugin for mocking.
    """
    ids = ["id1", "id2"]
    expected_results = [("doc1",), ("doc2",)]

    spy = mocker.spy(pgvector_service.engine, "connect")

    results = pgvector_service.get_by_ids(ids)

    assert results == expected_results
    spy.assert_called_once()


@pytest.mark.asyncio()
@pytest.mark.services()
@pytest.mark.pgvectoronly()
async def test_integration_get_all_by_collection_id(pgvector_service: PgvectorService, mocker: MockerFixture) -> None:
    """Test the get_all_by_collection_id method.

    Args:
        pgvector_service (PgvectorService): The PgvectorService instance.
        mocker (MockerFixture): The pytest-mock plugin for mocking.
    """
    collection_id = "test_collection_id"
    expected_results = [("doc1",), ("doc2",)]

    spy = mocker.spy(pgvector_service.engine, "connect")

    results = pgvector_service.get_all_by_collection_id(collection_id)

    assert results == expected_results
    spy.assert_called_once()
