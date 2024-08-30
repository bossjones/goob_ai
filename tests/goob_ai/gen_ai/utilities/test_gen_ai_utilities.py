"""
Tests for the gen_ai utilities module.

This module contains pytest tests for the functions in the gen_ai.utilities.__init__ module.
"""

from __future__ import annotations

import datetime

from collections.abc import Generator, Iterable, Iterator
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING, Optional, TypeVar
from uuid import UUID

from goob_ai.gen_ai.utilities import (
    add_chunk_id,
    add_item_checksum,
    add_item_last_seen_at,
    compute_hash,
    get_chunks_to_delete,
    get_chunks_to_update,
    get_dataset_loader,
    get_nested_value,
    stringify_dict,
)
from langchain.document_loaders import TextLoader
from langchain_core.documents import Document

import pytest


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch
    from vcr.request import Request as VCRRequest


def test_get_nested_value() -> None:
    """Test the get_nested_value function."""
    test_dict = {"a": "v1", "c1": {"c2": "v2"}}
    assert get_nested_value(test_dict, "a") == "v1"
    assert get_nested_value(test_dict, "c1.c2") == "v2"
    assert get_nested_value(test_dict, "nonexistent") == ""
    assert get_nested_value(test_dict, "c1.nonexistent") == ""


def test_stringify_dict() -> None:
    """Test the stringify_dict function."""
    test_dict = {"a": {"text": "Apify is cool"}, "description": "Apify platform"}
    result = stringify_dict(test_dict, ["a.text", "description"])
    assert result == "a.text: Apify is cool\ndescription: Apify platform"


def test_get_dataset_loader(mock_ebook_txt_file: FixtureRequest) -> None:
    """Test the get_dataset_loader function."""
    filename = f"{mock_ebook_txt_file}"
    loader = get_dataset_loader(filename)
    assert isinstance(loader, TextLoader)


def test_compute_hash() -> None:
    """Test the compute_hash function."""
    text = "Hello, World!"
    expected_hash = "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
    assert compute_hash(text) == expected_hash


def test_get_chunks_to_delete() -> None:
    """Test the get_chunks_to_delete function."""
    now = datetime.now(timezone.utc).timestamp()
    chunks_prev = [
        Document(page_content="Old1", metadata={"item_id": "1", "last_seen_at": int(now - 86400 * 2)}),
        Document(page_content="Old2", metadata={"item_id": "2", "last_seen_at": int(now - 86400 * 1)}),
    ]
    chunks_current = [Document(page_content="New", metadata={"item_id": "3"})]

    expired, keep = get_chunks_to_delete(chunks_prev, chunks_current, expired_days=1.5)
    assert len(expired) == 1
    assert len(keep) == 1
    assert expired[0].page_content == "Old1"
    assert keep[0].page_content == "Old2"


def test_get_chunks_to_update() -> None:
    """Test the get_chunks_to_update function."""
    chunks_prev = [
        Document(page_content="Old", metadata={"item_id": "1", "checksum": "abc"}),
        Document(page_content="Unchanged", metadata={"item_id": "2", "checksum": "def"}),
    ]
    chunks_current = [
        Document(page_content="New", metadata={"item_id": "1", "checksum": "xyz"}),
        Document(page_content="Unchanged", metadata={"item_id": "2", "checksum": "def"}),
        Document(page_content="Added", metadata={"item_id": "3", "checksum": "ghi"}),
    ]

    to_add, to_update = get_chunks_to_update(chunks_prev, chunks_current)
    assert len(to_add) == 2
    assert len(to_update) == 1
    assert to_add[0].page_content == "New"
    assert to_add[1].page_content == "Added"
    assert to_update[0].page_content == "Unchanged"


def test_add_item_last_seen_at() -> None:
    """Test the add_item_last_seen_at function."""
    items = [Document(page_content="Test", metadata={})]
    updated_items = add_item_last_seen_at(items)
    assert "last_seen_at" in updated_items[0].metadata
    assert isinstance(updated_items[0].metadata["last_seen_at"], int)


def test_add_item_checksum() -> None:
    """Test the add_item_checksum function."""
    items = [Document(page_content="Test", metadata={"key1": "value1", "key2": "value2"})]
    dataset_fields_to_item_id = ["key1", "key2"]
    updated_items = add_item_checksum(items, dataset_fields_to_item_id)
    assert "checksum" in updated_items[0].metadata
    assert "item_id" in updated_items[0].metadata
    assert isinstance(updated_items[0].metadata["checksum"], str)
    assert isinstance(updated_items[0].metadata["item_id"], str)


def test_add_chunk_id() -> None:
    """Test the add_chunk_id function."""
    chunks = [Document(page_content="Test", metadata={})]
    updated_chunks = add_chunk_id(chunks)
    assert "chunk_id" in updated_chunks[0].metadata
    assert UUID(updated_chunks[0].metadata["chunk_id"], version=4)


if __name__ == "__main__":
    pytest.main()
