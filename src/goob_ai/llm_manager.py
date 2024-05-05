from __future__ import annotations

import logging

from langchain.schema.runnable import ConfigurableField, Runnable, RunnableBranch, RunnableLambda, RunnableMap
from langchain_openai import ChatOpenAI
from loguru import logger as LOGGER
from pydantic import BaseModel

from goob_ai.aio_settings import aiosettings


class LlmManager(BaseModel):
    llm: ChatOpenAI | None = None

    def __init__(self):
        super().__init__()
        # SOURCE: https://github.com/langchain-ai/weblangchain/blob/main/main.py
        self.llm = ChatOpenAI(
            model=aiosettings.chat_model,
            streaming=True,
            temperature=aiosettings.llm_temperature,
        )
        # self.llm = ChatOpenAI(
        #     model="gpt-3.5-turbo-16k",
        #     # model="gpt-4",
        #     streaming=True,
        #     temperature=0.1,
        # ).configurable_alternatives(
        #     # This gives this field an id
        #     # When configuring the end runnable, we can then use this id to configure this field
        #     ConfigurableField(id="llm"),
        #     default_key="openai",
        #     anthropic=ChatAnthropic(
        #         model="claude-2",
        #         max_tokens=16384,
        #         temperature=0.1,
        #         anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "not_provided"),
        #     ),
        # )


class VisionModel(BaseModel):
    vision_api: ChatOpenAI | None = None

    def __init__(self):
        super().__init__()
        self.vision_api = ChatOpenAI(
            model=aiosettings.vision_model,
            max_tokens=900,
            temperature=aiosettings.llm_temperature,
        )

    # Pydantic doesn't seem to know the types to handle AzureOpenAI, so we need to tell it to allow arbitrary types
    class Config:
        arbitrary_types_allowed = True
