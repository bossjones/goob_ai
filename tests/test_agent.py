"""test_agent"""

# pylint: disable=assigning-non-slot
# pylint: disable=consider-using-from-import
from __future__ import annotations

from typing import TYPE_CHECKING

from goob_ai.agent import AiAgent

import pytest


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture

import asyncio
import sys
import uuid

from functools import partial
from itertools import cycle
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, List, Optional, Sequence, cast

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

import pytest


class AnyStr(str):
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, str)


def _with_nulled_run_id(events: Sequence[StreamEvent]) -> List[StreamEvent]:
    """Removes the run ids from events."""
    for event in events:
        assert "run_id" in event, f"Event {event} does not have a run_id."
        assert "parent_ids" in event, f"Event {event} does not have parent_ids."
        assert isinstance(event["run_id"], str), f"Event {event} run_id is not a string."
        assert isinstance(event["parent_ids"], list), f"Event {event} parent_ids is not a list."

    return cast(
        List[StreamEvent],
        [{**event, "run_id": "", "parent_ids": []} for event in events],
    )


async def _as_async_iterator(iterable: List) -> AsyncIterator:
    """Converts an iterable into an async iterator."""
    for item in iterable:
        yield item


async def _collect_events(events: AsyncIterator[StreamEvent], with_nulled_ids: bool = True) -> List[StreamEvent]:
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


@pytest.fixture
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


# @pytest.mark.parametrize(
#     "test_id, session_id, user_task, expected_output",
#     [
#         ("test_01", VALID_SESSION_ID, VALID_USER_TASK, "Expected response"),
#         ("test_02", VALID_SESSION_ID, INVALID_USER_TASK, "An error occurred while processing the task."),
#     ],
# )
# def test_process_user_task(
#     mocker: MockerFixture, monkeypatch: MonkeyPatch, agent: AiAgent, test_id, session_id, user_task, expected_output
# ):
#     monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_TOKEN", "fake_discord_token")
#     monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_TOKEN", "fake_discord_token")
#     monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_ADMIN_USER_ID", 1337)
#     monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_SERVER_ID", 1337)
#     monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_CLIENT_ID", 8008)
#     monkeypatch.setenv("GOOB_AI_CONFIG_OPENAI_API_KEY", "fake_openai_key")
#     monkeypatch.setenv("OPENAI_API_KEY", "fake_openai_key")
#     monkeypatch.setenv("PINECONE_API_KEY", "fake_pinecone_key")
#     monkeypatch.setenv("PINECONE_INDEX", "fake_test_index")
#     # Arrange
#     with mocker.patch.object(agent, "setup_agent_executor", return_value=mocker.MagicMock()) as mock_setup:
#         mock_executor = mock_setup.return_value
#         mock_executor.invoke.return_value = {"output": expected_output}

#     # Act
#     output = agent.process_user_task(session_id, user_task)

#     # Assert
#     assert output == expected_output
#     mock_setup.assert_called_once_with(session_id, user_task)
#     mock_executor.invoke.assert_called_once()


# @pytest.mark.parametrize(
#     "test_id, session_id, user_task, raises_exception, expected_exception",
#     [
#         ("test_03", VALID_SESSION_ID, VALID_USER_TASK, False, None),
#         ("test_04", VALID_SESSION_ID, "", True, ValueError),
#     ],
# )
# def test_process_user_task_exceptions(
#     mocker: MockerFixture,
#     monkeypatch: MonkeyPatch,
#     agent,
#     test_id,
#     session_id,
#     user_task,
#     raises_exception,
#     expected_exception,
# ):
#     monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_TOKEN", "fake_discord_token")
#     monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_TOKEN", "fake_discord_token")
#     monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_ADMIN_USER_ID", 1337)
#     monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_SERVER_ID", 1337)
#     monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_CLIENT_ID", 8008)
#     monkeypatch.setenv("GOOB_AI_CONFIG_OPENAI_API_KEY", "fake_openai_key")
#     monkeypatch.setenv("OPENAI_API_KEY", "fake_openai_key")
#     monkeypatch.setenv("PINECONE_API_KEY", "fake_pinecone_key")
#     monkeypatch.setenv("PINECONE_INDEX", "fake_test_index")
#     # Arrange
#     if raises_exception:
#         with mocker.patch.object(agent, "setup_agent_executor", side_effect=expected_exception, autospec=True):
#             # Act and Assert
#             with pytest.raises(expected_exception):
#                 agent.process_user_task(session_id, user_task)
#     else:
#         with mocker.patch.object(agent, "setup_agent_executor", return_value=mocker.MagicMock(), autospec=True) as mock_setup:
#             mock_executor = mock_setup.return_value
#             mock_executor.invoke.return_value = {"output": "Expected response"}

#             # Act
#             output = agent.process_user_task(session_id, user_task)

#             # Assert
#             assert output == "Expected response"
#             mock_setup.assert_called_once_with(session_id, user_task)
#             mock_executor.invoke.assert_called_once()


# Additional tests for init_agent_name, init_tools, init_agent_executor, process_user_task_streaming, and summarize would follow a similar pattern.
