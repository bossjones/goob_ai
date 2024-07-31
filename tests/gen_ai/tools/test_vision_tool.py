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


# def test_vision_tool_schema():
#     # Arrange
#     vision_tool = VisionTool()


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


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.visiontoolonly
# looks like pydantic models between patch versions of langchain can break things if you are not careful. Let's start writing some tests to ensure the schema remains the same between versions.
# @pytest.mark.parametrize(
#     "tool_",
#     [VisionTool()],
# )
def test_tool_injected_arg_with_schema(
    caplog,
    discord_image: FixtureRequest,
    vision_tool_prompt: FixtureRequest,
    monkeypatch: MonkeyPatch,
    mocker: MockerFixture,
    request: FixtureRequest,
) -> None:
    from goob_ai.gen_ai.tools import vision_tool
    # https://i.imgur.com/ae2d4hj.png
    # monkeypatch.setattr(vision_tool, "DISCORD_URL_PATTERN", r"https?://i\.imgur\.com/.*")

    tool_ = vision_tool.VisionTool(
        image_path=discord_image,
        prompt=vision_tool_prompt,
    )
    assert tool_.get_input_schema().schema() == {
        "title": "VisionToolInput",
        "description": "Use this tool to get more information about an image given a URL to an image file. Use for all urls for image files including discord urls.\n\nArgs:\n    image_path: The URL to the image file\n    prompt: The prompt to use for the API call\n\nReturns: The response from the Vision API",
        "type": "object",
        "properties": {
            "image_path": {"title": "Image Path", "description": "The URL to the image file.", "type": "string"},
            "prompt": {"title": "Prompt", "description": "The prompt to use for the API call.", "type": "string"},
        },
        "required": ["image_path", "prompt"],
    }

    assert tool_.args_schema.schema() == {
        "title": "VisionToolInput",
        "description": "Use this tool to get more information about an image given a URL to an image file. Use for all urls for image files including discord urls.\n\nArgs:\n    image_path: The URL to the image file\n    prompt: The prompt to use for the API call\n\nReturns: The response from the Vision API",
        "type": "object",
        "properties": {
            "image_path": {"title": "Image Path", "description": "The URL to the image file.", "type": "string"},
            "prompt": {"title": "Prompt", "description": "The prompt to use for the API call.", "type": "string"},
        },
        "required": ["image_path", "prompt"],
    }

    # res = tool_._run(
    #     discord_image,
    #     vision_tool_prompt,
    #     config={"tags": ["pytest", "ci", "synchronous"]},
    # )

    # # assert "Ego" in res
    # # assert "@HowFarCanWeFall" in res
    # assert "Too fine. Blocked" in res

    # validation_error = ValidationError(model='VisionToolInput', errors=[{'loc': ('image_path',), 'msg': 'field required', 'type': 'value_error.missing'}, {'loc': ('prompt',), 'msg': 'field required', 'type': 'value_error.missing'}])

    # E   pydantic.v1.error_wrappers.ValidationError: 2 validation errors for VisionToolInput
    # E   image_path
    # E     field required (type=value_error.missing)
    # E   prompt
    # E     field required (type=value_error.missing)
    # expected_error = ValidationError if not isinstance(tool_, vision_tool.VisionTool) else TypeError
    # with pytest.raises(
    #     ValidationError,
    #     match=r".*(2 validation errors for VisionToolInput|Error in read_image_tool: Connection error).*",
    # ) as excinfo:
    #     tool_.invoke({"x": 5})

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


# # NOTE: looks like pydantic models between patch versions of langchain can break things if you are not careful. Let's start writing some tests to ensure the schema remains the same between versions.
# @pytest.mark.parametrize(
#     "tool_",
#     [vision_tool.VisionTool()],
# )
# @pytest.mark.visiontoolonly
# @pytest.mark.asyncio
# async def test_async_vision_tool_injected_arg_with_schema(
#     tool_: BaseTool,
#     caplog,
#     slack_image: FixtureRequest,
#     vision_tool_prompt: FixtureRequest,
#     monkeypatch: MonkeyPatch,
#     mocker: MockerFixture,
#     request: FixtureRequest,
# ) -> None:
#     # this normally gets set inside of agent.py, set it here so we mimick behavior.
#     tool_.handle_tool_error = True

#     assert tool_.get_input_schema().schema() == {
#         "title": "VisionToolInput",
#         "description": "Use this tool to get more information about an image given a URL to an image file. Use for all urls for image files including slack urls.\n\nArgs:\n    image_path: The URL to the image file\n    prompt: The prompt to use for the API call",
#         "type": "object",
#         "properties": {
#             "image_path": {"title": "Image Path", "description": "The URL to the image file.", "type": "string"},
#             "prompt": {"title": "Prompt", "description": "The prompt to use for the API call.", "type": "string"},
#         },
#         "required": ["image_path", "prompt"],
#     }

#     # NOTE: interestingly, the title is different when you call args_schema.schema() vs get_input_schema().schema()
#     assert tool_.args_schema.schema() == {
#         "title": "VisionToolInput",
#         "description": "Use this tool to get more information about an image given a URL to an image file. Use for all urls for image files including slack urls.\n\nArgs:\n    image_path: The URL to the image file\n    prompt: The prompt to use for the API call",
#         "type": "object",
#         "properties": {
#             "image_path": {"title": "Image Path", "description": "The URL to the image file.", "type": "string"},
#             "prompt": {"title": "Prompt", "description": "The prompt to use for the API call.", "type": "string"},
#         },
#         "required": ["image_path", "prompt"],
#     }

#     # validate object properites
#     assert tool_.name == "vision_api"
#     assert (
#         tool_.description
#         == "Use this tool to get more information about an image given a URL to an image file. Use for all urls for image files including slack urls.\n    Args:\n        image_path: The URL to the image file\n        prompt: The prompt to use for the API call\n"
#     )

#     # NOTE: this does not seem like a sufficent docstring, but it is what is in the code.
#     assert tool_.__doc__ == "Tool that can operate on any number of inputs."

#     # validate the tool properties via dict
#     tool_properties = tool_.dict()
#     assert tool_properties["name"] == "vision_api"
#     assert (
#         tool_properties["description"]
#         == "Use this tool to get more information about an image given a URL to an image file. Use for all urls for image files including slack urls.\n    Args:\n        image_path: The URL to the image file\n        prompt: The prompt to use for the API call\n"
#     )
#     assert tool_properties["return_direct"] == False
#     assert tool_properties["handle_tool_error"]
#     assert tool_properties["handle_validation_error"] == False

#     # integration test
#     res = await tool_._arun(
#         slack_image,
#         vision_tool_prompt,
#         config={"tags": ["pytest", "ci", "async"]},
#     )

#     # Define the partial regex pattern to search for
#     pattern = r".*code.*7"
#     # Use re.search to check for a partial match
#     match = re.search(pattern, res)
#     # Assert that a match is found
#     assert match is not None

#     # assert '"code":7' in res
#     assert "workflows.argoproj.io is forbidden" in res
#     assert "system:serviceaccount:argo:user-default-login" in str(res)
#     assert "ns-team-cc-education--edu-sam-deploy--c407667f--names-e09b2293" in str(res)

#     with pytest.raises(
#         TypeError,
#         match=r".*VisionTool._run.*missing 2 required positional arguments.*'image_path' and 'prompt'.*",
#     ) as excinfo:
#         await tool_.ainvoke({"x": 5}, config={"tags": ["pytest", "ci", "async"]})

#     assert convert_to_openai_function(tool_) == {
#         "name": "vision_api",
#         "description": "Use this tool to get more information about an image given a URL to an image file. Use for all urls for image files including slack urls.\n\nArgs:\n    image_path: The URL to the image file\n    prompt: The prompt to use for the API call\n\nReturns: The response from the Vision API",
#         "parameters": {
#             "type": "object",
#             "properties": {"image_path": {"type": "string"}, "prompt": {"type": "string"}},
#             "required": ["image_path", "prompt"],
#         },
#     }
