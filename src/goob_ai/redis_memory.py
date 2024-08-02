# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
# pylint: disable=no-name-in-module

"""redis_memory.py"""

from __future__ import annotations

import logging

from typing import Any, Dict, List, Optional, Tuple

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ReadOnlySharedMemory
from langchain.schema import AIMessage, HumanMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory
from loguru import logger as LOGGER

from goob_ai.aio_settings import aiosettings


# SOURCE: https://github.com/chatcrypto/chatcrypto-public/blob/034785510e49cf72b87c249ca90201ef483d754d/chatcryptor/bot/base/base_memory.py#L27
class CustomRedisChatMessageHistory(RedisChatMessageHistory):
    """Chat message history stored in a Redis database."""

    def add_ai_message(self, message: str) -> None:
        """
        HACK: Disable ai message for memory
        """
        return True


# SOURCE: https://github.com/chatcrypto/chatcrypto-public/blob/034785510e49cf72b87c249ca90201ef483d754d/chatcryptor/bot/base/base_memory.py#L27
class CustomConversationBufferMemory(ConversationBufferMemory):
    def _get_input_output(self, inputs: dict[str, Any], outputs: dict[str, str]) -> tuple[str, str]:
        # HACK: because key `groups`` is used for filter which tools that are allowed to execute,
        # but its conflict with `inputs` of memory, so will be removed from inputs.
        if "groups" in inputs:
            del inputs["groups"]
        return super()._get_input_output(inputs, outputs)


# SOURCE: https://github.com/robertdowneyjnr/db-chatbot/blob/d2a7224b93521516cc085a5001d5a9ee315a88dd/ui.py#L6
# "foo", url=f"{aiosettings.redis_url}", key_prefix="goob:"
class RedisMemory:
    def __init__(self, redis_url: str | None, session_id: str | None):
        if not redis_url:
            self.redis_url = f"{aiosettings.redis_url}"
        self.redis_url = redis_url
        self.session_id = session_id
        try:
            self.history = RedisChatMessageHistory(url=self.redis_url, session_id=self.session_id)
        except Exception as e:
            LOGGER.error(f"Failed to initialize RedisChatMessageHistory: {e}")
            self.history = None

    def add_message(self, role, content):
        if self.history is None:
            LOGGER.error("RedisChatMessageHistory is not initialized")
            return

        if role == "human":
            message = HumanMessage(content=content)
        elif role == "ai":
            message = AIMessage(content=content)
        else:
            raise ValueError(f"Unsupported role: {role}")

        try:
            self.history.add_message(message)
        except Exception as e:
            LOGGER.error(f"Failed to add message to Redis: {e}")

    def get_messages(self):
        if self.history is None:
            LOGGER.error("RedisChatMessageHistory is not initialized")
            return []

        try:
            return self.history.messages
        except Exception as e:
            LOGGER.error(f"Failed to get messages from Redis: {e}")
            return []

    def clear(self):
        if self.history is None:
            LOGGER.error("RedisChatMessageHistory is not initialized")
            return

        try:
            self.history.clear()
        except Exception as e:
            LOGGER.error(f"Failed to clear Redis history: {e}")

    @staticmethod
    def generate_session_id(question):
        import hashlib

        session_id = f"session:{hashlib.md5(question.encode()).hexdigest()[:10]}"
        LOGGER.debug(f"Session id: {session_id}")
        return session_id

    @staticmethod
    def generate_session_name(question):
        # Generate a session name based on the question (e.g., first few words)
        session_name = " ".join(question.split()[:5])
        LOGGER.debug(f"Session name: {session_name}")
        return session_name


# https://github.com/chatcrypto/chatcrypto-public/blob/034785510e49cf72b87c249ca90201ef483d754d/chatcryptor/bot/base/base_memory.py#L27
def create_memory_instance(session_id: str = "", store_ai_answer=False, ttl=60, **kwargs):
    """
    Create base memory instance that used for all agents, chains.
    """
    if session_id.strip() == "":
        raise ValueError("session_id must be set")
    LOGGER.debug(f"Start create memory instance with Redis and session_id: {session_id}")
    if store_ai_answer is False:
        LOGGER.error("Using regular redis memory")
        message_history = RedisMemory(f"{aiosettings.redis_url}", session_id)
    else:
        LOGGER.error("Using experimental redis memory")
        message_history = RedisChatMessageHistory(session_id, url=f"{aiosettings.redis_url}", ttl=ttl)
    # memory = CustomConversationBufferMemory(
    #     memory_key="chat_history", chat_memory=message_history, return_messages=True, k=aiosettings.chat_history_buffer,
    #         output_key="output",
    # )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        return_messages=True,
        k=aiosettings.chat_history_buffer,
        output_key="output",
    )
    return memory


def create_readonly_memory(memory=None):
    if memory:
        return ReadOnlySharedMemory(memory=memory)
    return None
