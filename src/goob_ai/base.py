# via gpt-discord-bot
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


SEPARATOR_TOKEN = "<|endoftext|>"

###################################################################
# NOTE: on frozen dataclasses
###################################################################
# Frozen instances
# It is not possible to create truly immutable Python objects. However, by passing frozen=True to the @dataclass decorator you can emulate immutability. In that case, dataclasses will add __setattr__() and __delattr__() methods to the class. These methods will raise a FrozenInstanceError when invoked.

# There is a tiny performance penalty when using frozen=True: __init__() cannot use simple assignment to initialize fields, and must use object.__setattr__().
###################################################################


@dataclass(frozen=True)
class GoobMessage:
    user: str
    text: Optional[str] = None

    def render(self):
        result = self.user + ":"
        if self.text is not None:
            result += " " + self.text
        return result


@dataclass
class GoobConversation:
    messages: List[GoobMessage]

    def prepend(self, message: GoobMessage):
        self.messages.insert(0, message)
        return self

    def render(self):
        return f"\n{SEPARATOR_TOKEN}".join([message.render() for message in self.messages])


@dataclass(frozen=True)
class GoobConfig:
    name: str
    instructions: str
    example_conversations: List[GoobConversation]


@dataclass(frozen=True)
class GoobThreadConfig:
    model: str
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class GoobPrompt:
    header: GoobMessage
    examples: List[GoobConversation]
    convo: GoobConversation

    def full_render(self, bot_name: str):
        messages = [
            {
                "role": "system",
                "content": self.render_system_prompt(),
            }
        ]
        for message in self.render_messages(bot_name):
            messages.append(message)
        return messages

    def render_system_prompt(self):
        return f"\n{SEPARATOR_TOKEN}".join(
            [self.header.render()]
            + [GoobMessage("System", "Example conversations:").render()]
            + [conversation.render() for conversation in self.examples]
            + [GoobMessage("System", "Now, you will work with the actual current conversation.").render()]
        )

    def render_messages(self, bot_name: str):
        for message in self.convo.messages:
            if not bot_name in message.user:
                yield {
                    "role": "user",
                    "name": message.user,
                    "content": message.text,
                }
            else:
                yield {
                    "role": "assistant",
                    "name": bot_name,
                    "content": message.text,
                }
