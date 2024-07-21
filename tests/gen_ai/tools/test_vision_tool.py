from __future__ import annotations

import inspect
import json
import logging
import os.path
import sys
import textwrap

from datetime import datetime
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Type, Union

from goob_ai.gen_ai.tools.vision_tool import DISCORD_URL_PATTERN, VisionTool
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
from typing_extensions import Annotated, TypedDict

import pytest


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture

logger = logging.getLogger(__name__)


def test_vision_tool_schema():
    # Arrange
    vision_tool = VisionTool()


@pytest.fixture
def discord_image() -> str:
    return "https://i.imgur.com/ae2d4hj.png"
    # return "http://127.0.0.1:19000/vision_api_fixture.PNG"
    # return "https://cdn.discordapp.com/attachments/1237526936201334804/1264394713868406845/20240720_1_screenshot_image_larger00009.PNG?ex=669db6d7&is=669c6557&hm=d1bfe1131eb4938949ca4645de73ee51a36aca7d2ef8cbb83f3b3c7bb33eab57&"


@pytest.fixture
def vision_tool_prompt() -> str:
    return "Extract text from the image"


# https://smith.langchain.com/o/631f824f-4072-5bc6-b1f6-924eb5dfd83f/projects/p/9c97a8d8-3b8b-4b42-85be-6a17c4aab36d?timeModel=%7B%22duration%22%3A%2214d%22%7D&searchModel=%7B%22filter%22%3A%22eq%28run_type%2C+%5C%22tool%5C%22%29%22%2C%22traceFilter%22%3A%22%22%7D&runtab=0&peek=e10d70e0-ba49-4ed7-874f-ef5caaac2d3c
# input: "{'image_path': 'https://cdn.discordapp.com/attachments/1237526936201334804/1264394713868406845/20240720_1_screenshot_image_larger00009.PNG?ex=669db6d7&is=669c6557&hm=d1bfe1131eb4938949ca4645de73ee51a36aca7d2ef8cbb83f3b3c7bb33eab57&', 'prompt': 'Extract text from the image'}"
@pytest.mark.visiontoolonly
# looks like pydantic models between patch versions of langchain can break things if you are not careful. Let's start writing some tests to ensure the schema remains the same between versions.
# @pytest.mark.parametrize(
#     "tool_",
#     [VisionTool()],
# )
def test_tool_injected_arg_with_schema(
    caplog, discord_image: FixtureRequest, vision_tool_prompt: FixtureRequest
) -> None:
    tool_ = VisionTool(
        image_path=discord_image,
        prompt=vision_tool_prompt,
    )
    assert tool_.get_input_schema().schema() == {
        "title": "VisionToolInput",
        "type": "object",
        "properties": {
            "image_path": {"title": "Image Path", "description": "The URL to the image file.", "type": "string"},
            "prompt": {"title": "Prompt", "description": "The prompt to use for the API call.", "type": "string"},
        },
        "required": ["image_path", "prompt"],
    }

    assert tool_.args_schema.schema() == {
        "title": "VisionToolInput",
        "type": "object",
        "properties": {
            "image_path": {"title": "Image Path", "description": "The URL to the image file.", "type": "string"},
            "prompt": {"title": "Prompt", "description": "The prompt to use for the API call.", "type": "string"},
        },
        "required": ["image_path", "prompt"],
    }
    res = tool_.invoke(
        {
            "image_path": discord_image,
            "prompt": vision_tool_prompt,
        }
    )

    assert "Ego" in res
    assert "@HowFarCanWeFall" in res
    assert "Too fine. Blocked" in res

    # validation_error = ValidationError(model='VisionToolInput', errors=[{'loc': ('image_path',), 'msg': 'field required', 'type': 'value_error.missing'}, {'loc': ('prompt',), 'msg': 'field required', 'type': 'value_error.missing'}])

    # E   pydantic.v1.error_wrappers.ValidationError: 2 validation errors for VisionToolInput
    # E   image_path
    # E     field required (type=value_error.missing)
    # E   prompt
    # E     field required (type=value_error.missing)
    expected_error = ValidationError if not isinstance(tool_, VisionTool) else TypeError
    with pytest.raises(
        ValidationError,
        match=r".*(2 validation errors for VisionToolInput|Error in read_image_tool: Connection error).*",
    ) as excinfo:
        tool_.invoke({"x": 5})

    assert convert_to_openai_function(tool_) == {
        "name": "vision_api",
        "description": "This tool calls OpenAI's Vision API to get more information about an image given a URL to an image file.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"description": "The URL to the image file.", "type": "string"},
                "prompt": {"description": "The prompt to use for the API call.", "type": "string"},
            },
            "required": ["image_path", "prompt"],
        },
    }
