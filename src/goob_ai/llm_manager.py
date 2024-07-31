# pylint: disable=no-member
# sourcery skip: docstrings-for-classes
from __future__ import annotations

import logging

import openai

from langchain.pydantic_v1 import BaseModel
from langchain.schema.runnable import ConfigurableField, Runnable, RunnableBranch, RunnableLambda, RunnableMap

# from openai import Client
from langchain_openai import ChatOpenAI, OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from loguru import logger as LOGGER

from goob_ai.aio_settings import aiosettings


# NOTE: FIXME: Set these model settings https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
# NOTE: FIXME: Set these model settings https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
# NOTE: FIXME: Set these model settings https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
# NOTE: FIXME: Set these model settings https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
# NOTE: FIXME: Set these model settings https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

# def get_supported_openai_params(self):
#     return [
#         "stream",
#         "temperature",
#         "top_p",
#         "max_tokens",
#         "tools",
#         "tool_choice",
#         "seed",
#         "response_format",
#     ]


MODELS_MAP = {
    "gpt-4o": {
        "params": {
            "temperature": 0.0,
            # This optional parameter helps to set the maximum number of tokens to generate in the chat completion.
            "max_tokens": 4096,
            # "max_input_tokens": 128000,
            # "max_output_tokens": 4096,
        },
    },
    "gpt-4-turbo": {
        "params": {
            "max_tokens": 4096,
            # "max_input_tokens": 128000,
            # "max_output_tokens": 4096,
            # "temperature": 0.0,
        },
    },
    "gpt-4": {
        "params": {
            "temperature": 0,
            "max_tokens": 4096,
            # "max_input_tokens": 8192,
            # "max_output_tokens": 4096,
        },
    },
    "gpt-3.5-turbo": {
        "params": {
            "temperature": 0,
        },
    },
    "gemma-7b-it": {
        "params": {
            "temperature": 0,
        },
    },
    "gemma2-9b-it": {
        "params": {
            "temperature": 0,
        },
    },
    "llama3-70b-8192": {
        "params": {
            "temperature": 0,
        },
    },
    "llama3-8b-8192": {
        "params": {
            "temperature": 0,
        },
    },
    "mixtral-8x7b-32768": {
        "params": {
            "temperature": 0,
        },
    },
    "claude-3-haiku-20240307": {
        "params": {
            # "temperature": 0,
            "max_tokens": 4096,
            # "max_input_tokens": 200000,
            # "max_output_tokens": 4096,
        },
    },
    "claude-3-opus-20240229": {
        "params": {
            # "temperature": 0,
            "max_tokens": 4096,
            # "max_input_tokens": 200000,
            # "max_output_tokens": 4096,
        },
    },
    "claude-3-sonnet-20240229": {
        "params": {
            # "temperature": 0,
            "max_tokens": 4096,
            # "max_input_tokens": 200000,
            # "max_output_tokens": 4096,
        },
    },
    "claude-3-5-sonnet-20240620": {
        "params": {
            # "temperature": 0,
            "max_tokens": 4096,
            # "max_input_tokens": 200000,
            # "max_output_tokens": 4096,
        },
    },
    "anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "params": {
            # "temperature": 0,
            "max_tokens": 4096,
            # "max_input_tokens": 200000,
            # "max_output_tokens": 4096,
        },
    },
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "params": {
            # "temperature": 0,
            "max_tokens": 4096,
            # "max_input_tokens": 200000,
            # "max_output_tokens": 4096,
        },
    },
    "anthropic.claude-3-opus-20240229-v1:0": {
        "params": {
            # "temperature": 0,
            "max_tokens": 4096,
            # "max_input_tokens": 200000,
            # "max_output_tokens": 4096,
        },
    },
}


class LlmManager(BaseModel):
    llm: ChatOpenAI | None = None

    def __init__(self):
        super().__init__()
        # SOURCE: https://github.com/langchain-ai/weblangchain/blob/main/main.py
        self.llm = ChatOpenAI(
            name="ChatOpenAI",
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
            max_retries=5,
            max_tokens=900,
            temperature=0.0,
        )
        # self.vision_api = wrap_openai(Client(api_key=aiosettings.openai_api_key.get_secret_value()))

    # Pydantic doesn't seem to know the types to handle AzureOpenAI, so we need to tell it to allow arbitrary types
    class Config:
        arbitrary_types_allowed = True
