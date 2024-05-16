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
