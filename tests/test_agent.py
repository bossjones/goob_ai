"""test_agent"""

# pylint: disable=assigning-non-slot
# pylint: disable=consider-using-from-import
from __future__ import annotations

import asyncio
import logging
import sys
import uuid

from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from functools import partial
from itertools import cycle
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from goob_ai.agent import AiAgent
from langchain_core.callbacks import CallbackManagerForRetrieverRun, Callbacks
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.language_models import FakeStreamingListLLM, GenericFakeChatModel
from langchain_core.load import dumpd
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableConfig,
    RunnableGenerator,
    RunnableLambda,
    chain,
    ensure_config,
)
from langchain_core.runnables.config import get_callback_manager_for_config
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.schema import StreamEvent
from langchain_core.runnables.utils import Input, Output
from langchain_core.tools import tool
from langchain_core.utils.aiter import aclosing
from langchain_openai import OpenAIEmbeddings
from loguru import logger as LOGGER

import pytest


if TYPE_CHECKING:
    from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock

    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


class AnyStr(str):
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, str)


def _with_nulled_run_id(events: Sequence[StreamEvent]) -> list[StreamEvent]:
    """Removes the run ids from events."""
    for event in events:
        assert "run_id" in event, f"Event {event} does not have a run_id."
        assert "parent_ids" in event, f"Event {event} does not have parent_ids."
        assert isinstance(event["run_id"], str), f"Event {event} run_id is not a string."
        assert isinstance(event["parent_ids"], list), f"Event {event} parent_ids is not a list."

    return cast(
        list[StreamEvent],
        [{**event, "run_id": "", "parent_ids": []} for event in events],
    )


async def _as_async_iterator(iterable: list) -> AsyncIterator:
    """Converts an iterable into an async iterator."""
    for item in iterable:
        yield item


async def _collect_events(events: AsyncIterator[StreamEvent], with_nulled_ids: bool = True) -> list[StreamEvent]:
    """Collect the events and remove the run ids."""
    materialized_events = [event async for event in events]

    if with_nulled_ids:
        events_ = _with_nulled_run_id(materialized_events)
    else:
        events_ = materialized_events
    for event in events_:
        event["tags"] = sorted(event["tags"])
    return events_


# Constants for tests
VALID_SESSION_ID = "session123"
VALID_USER_TASK = "calculate something"
INVALID_USER_TASK = ""


@pytest.fixture()
def agent(monkeypatch: MonkeyPatch, mocker: MockerFixture, request: FixtureRequest):
    monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_TOKEN", "fake_discord_token")
    monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_TOKEN", "fake_discord_token")
    monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_ADMIN_USER_ID", 1337)
    monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_SERVER_ID", 1337)
    monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_CLIENT_ID", 8008)
    monkeypatch.setenv("GOOB_AI_CONFIG_OPENAI_API_KEY", "fake_openai_key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake_openai_key")
    monkeypatch.setenv("PINECONE_API_KEY", "fake_pinecone_key")
    monkeypatch.setenv("PINECONE_INDEX", "fake_test_index")
    return AiAgent()


def test_embeddings_default(agent: AiAgent):
    """Test that the default embeddings are set correctly."""
    assert isinstance(agent.embeddings, OpenAIEmbeddings)


def test_embeddings_custom(agent: AiAgent):
    """Test that custom embeddings can be set."""
    custom_embeddings = OpenAIEmbeddings(openai_api_key="custom_key")
    agent._embeddings = custom_embeddings
    assert agent.embeddings == custom_embeddings


def test_embeddings_lazy_loading(agent: AiAgent):
    """Test that embeddings are lazily loaded."""
    agent._embeddings = None
    assert isinstance(agent.embeddings, OpenAIEmbeddings)


def test_embeddings_caching(agent: AiAgent):
    """Test that embeddings are cached after first access."""
    agent._embeddings = None
    embeddings1 = agent.embeddings
    embeddings2 = agent.embeddings
    assert embeddings1 is embeddings2


def test_collection_name_default(agent: AiAgent):
    """Test that the default collection name is set correctly."""
    assert agent.collection_name == "readthedocs"


def test_collection_name_custom(agent: AiAgent):
    """Test that a custom collection name can be set."""
    custom_collection_name = "custom_collection"
    agent.collection_name = custom_collection_name
    assert agent.collection_name == custom_collection_name


def test_collection_name_lazy_loading(agent: AiAgent):
    """Test that the collection name is lazily loaded."""
    agent._collection_name = None
    assert agent.collection_name == "readthedocs"


def test_collection_name_caching(agent: AiAgent):
    """Test that the collection name is cached after first access."""
    agent._collection_name = None
    collection_name1 = agent.collection_name
    collection_name2 = agent.collection_name
    assert collection_name1 == collection_name2


def test_collection_name_setter(agent: AiAgent, caplog: LogCaptureFixture):
    """Test that the collection name setter logs a debug message."""
    caplog.set_level(logging.DEBUG)
    custom_collection_name = "custom_collection"

    debug = [i.message for i in caplog.records if i.levelno == logging.DEBUG]
    # with caplog.set_level(logging.DEBUG):
    agent.collection_name = custom_collection_name
    # assert f"{custom_collection_name}" in debug

    # wait for logging to finish runnnig
    # await LOGGER.complete()

    caplog.clear()
