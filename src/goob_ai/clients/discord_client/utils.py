# pylint: disable=too-many-function-args
# mypy: disable-error-code="arg-type, var-annotated, list-item, no-redef, truthy-bool, return-value"
# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import logging

from typing import List, Optional

import discord

from discord import Message as DiscordMessage
from loguru import logger as LOGGER

from goob_ai.base import GoobMessage
from goob_ai.constants import INACTIVATE_THREAD_PREFIX, MAX_CHARS_PER_REPLY_MSG


def discord_message_to_message(message: DiscordMessage) -> Optional[GoobMessage]:
    if (
        message.type == discord.MessageType.thread_starter_message
        and message.reference.cached_message
        and len(message.reference.cached_message.embeds) > 0
        and len(message.reference.cached_message.embeds[0].fields) > 0
    ):
        field = message.reference.cached_message.embeds[0].fields[0]
        if field.value:
            return GoobMessage(user=field.name, text=field.value)
    else:
        if message.content:
            return GoobMessage(user=message.author.name, text=message.content)
    return None


def split_into_shorter_messages(message: str) -> List[str]:
    return [message[i : i + MAX_CHARS_PER_REPLY_MSG] for i in range(0, len(message), MAX_CHARS_PER_REPLY_MSG)]


def is_last_message_stale(interaction_message: DiscordMessage, last_message: DiscordMessage, bot_id: str) -> bool:
    return (
        last_message
        and last_message.id != interaction_message.id
        and last_message.author
        and last_message.author.id != bot_id
    )


async def close_thread(thread: discord.Thread):
    await thread.edit(name=INACTIVATE_THREAD_PREFIX)
    await thread.send(
        embed=discord.Embed(
            description="**Thread closed** - Context limit reached, closing...",
            color=discord.Color.blue(),
        )
    )
    await thread.edit(archived=True, locked=True)


# def should_block(guild: Optional[discord.Guild]) -> bool:
#     if guild is None:
#         # dm's not supported
#         LOGGER.info(f"DM not supported")
#         return True

#     if guild.id and guild.id not in ALLOWED_SERVER_IDS:
#         # not allowed in this server
#         LOGGER.info(f"Guild {guild} not allowed")
#         return True
#     return False
