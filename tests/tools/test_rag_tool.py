from __future__ import annotations

import inspect
import json
import logging
import os.path
import sys
import textwrap

from collections.abc import Iterable, Iterator
from datetime import datetime
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Annotated, Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.messages import ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, ensure_config
from langchain_core.tools import (
    BaseTool,
    SchemaAnnotationError,
    StructuredTool,
    Tool,
    ToolException,
    _create_subset_model,
    tool,
)
from langchain_core.utils.function_calling import convert_to_openai_function
from loguru import logger as LOGGER
from requests_mock.mocker import Mocker as RequestsMocker
from typing_extensions import TypedDict

import pytest


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture

logger = logging.getLogger(__name__)


@pytest.fixture()
def rag_tool_prompt() -> str:
    # return "Using readthedocs, given the following text 'alert! Something Happened', how do I rich.print 'alert!' with style 'bold red; using rich.print? Do not use 'rich.console' in your answer."
    return "Using readthedocs, What the minimum version of python needed to install rich?"


@pytest.mark.visiontoolonly()
def test_rag_tool_injected_arg_with_schema(
    caplog,
    rag_tool_prompt: FixtureRequest,
    monkeypatch: MonkeyPatch,
    mocker: MockerFixture,
    request: FixtureRequest,
) -> None:
    # initalize the tool
    from goob_ai import llm_manager
    from goob_ai.aio_settings import AioSettings, aiosettings
    from goob_ai.gen_ai.tools.vision_tool import VisionTool
    from goob_ai.llm_manager import LlmManager
    from goob_ai.services.chroma_service import ChromaService
    from goob_ai.tools import rag_tool
    from goob_ai.tools.rag_tool import ReadTheDocsQATool
    from langchain_chroma import Chroma
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from pydantic_settings import SettingsConfigDict

    embeddings = OpenAIEmbeddings()
    db = Chroma(
        client=ChromaService.client,
        collection_name="readthedocs",
        embedding_function=embeddings,
    )
    llm = llm_manager.LlmManager().llm
    rtd_tool = ReadTheDocsQATool(db=db, llm=llm)

    tool_ = rtd_tool

    assert tool_.name == "chroma_question_answering"
    assert (
        tool_.description
        == "You must use this tool for any questions or queries related to opencv, rich, and Pillow or substrings of it. This will return documents that are related to the user's question. The documents may not be always relevant to the user's question. If you use any of the documents returned to provide a helpful answer to question, please make sure to also return a valid URL of the document you used."
    )
    assert str(tool_.args_schema) == "<class 'goob_ai.tools.rag_tool.ReadTheDocsQASchema'>"
    assert tool_.return_direct == False
    assert tool_.verbose == False
    assert tool_.tags == None
    assert tool_.metadata == None
    assert tool_.handle_tool_error == False
    assert tool_.handle_validation_error == False

    assert tool_.get_input_schema().schema() == {
        "title": "ReadTheDocsQASchema",
        "description": "You must use this tool for any questions or queries related to opencv, rich, and Pillow or substrings of it. This will return documents that are related to the user's question. The documents may not be always relevant to the user's question. If you use any of the documents returned to provide a helpful answer to question, please make sure to also return a valid URL of the document you used.\n\nArgs:\n----\n    question: A question to ask about a readthedocs pdf. Cannot be empty. Must be a question abount opencv, rich, or Pillow.",
        "type": "object",
        "properties": {
            "question": {
                "title": "Question",
                "description": "A question to ask about a readthedocs pdf. Cannot be empty. Must be a question abount opencv, rich, or Pillow.",
                "type": "string",
            }
        },
        "required": ["question"],
    }

    assert tool_.args_schema.schema() == {
        "title": "ReadTheDocsQASchema",
        "description": "You must use this tool for any questions or queries related to opencv, rich, and Pillow or substrings of it. This will return documents that are related to the user's question. The documents may not be always relevant to the user's question. If you use any of the documents returned to provide a helpful answer to question, please make sure to also return a valid URL of the document you used.\n\nArgs:\n----\n    question: A question to ask about a readthedocs pdf. Cannot be empty. Must be a question abount opencv, rich, or Pillow.",
        "type": "object",
        "properties": {
            "question": {
                "title": "Question",
                "description": "A question to ask about a readthedocs pdf. Cannot be empty. Must be a question abount opencv, rich, or Pillow.",
                "type": "string",
            }
        },
        "required": ["question"],
    }

    assert convert_to_openai_function(tool_) == {
        "name": "chroma_question_answering",
        "description": "You must use this tool for any questions or queries related to opencv, rich, and Pillow or substrings of it. This will return documents that are related to the user's question. The documents may not be always relevant to the user's question. If you use any of the documents returned to provide a helpful answer to question, please make sure to also return a valid URL of the document you used.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "description": "A question to ask about a readthedocs pdf. Cannot be empty. Must be a question abount opencv, rich, or Pillow.",
                    "type": "string",
                }
            },
            "required": ["question"],
        },
    }
