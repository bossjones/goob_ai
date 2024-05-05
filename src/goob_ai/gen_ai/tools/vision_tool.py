from __future__ import annotations

import base64
import logging
import re
import sys
import uuid

from typing import ClassVar, Dict, Optional, Type

import requests


# from clients.http_client import HttpClient
# from config import FlexChecksSettings
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.tools import ToolException

# LOGGER = logging.getLogger(__name__)
from loguru import logger as LOGGER

from goob_ai.aio_settings import aiosettings
from goob_ai.clients.http_client import HttpClient
from goob_ai.llm_manager import VisionModel


class VisionToolInput(BaseModel):
    image_path: str = Field(description="The URL to the image file.")
    prompt: str = Field(description="The prompt to use for the API call.")


class VisionTool(BaseTool):
    name = "vision_api"
    description = (
        "This tool calls OpenAI's Vision API to get more information about an image given a URL to an image file."
    )
    args_schema: Type[BaseModel] = VisionToolInput
    return_direct: bool = False
    handle_tool_error: bool = True

    def _run(
        self,
        image_path: str,
        prompt: str,
    ) -> str:
        """
        Use this tool to get more information about an image given a URL to an image file. Use for all urls for image files including discord urls.

        Args:
            image_path: The URL to the image file
            prompt: The prompt to use for the API call

        Returns: The response from the Vision API
        """
        try:
            # Initialize the Vision API client
            client = VisionModel().vision_api

            # Initialize the discord settings
            discord_token = aiosettings.discord_token

            # Function to download image from discord and convert to base64
            def fetch_image_from_discord(url: str) -> str:
                headers = {"Authorization": f"Bearer {discord_token}"}

                try:
                    response: requests.Response = HttpClient().get(url, headers=headers)
                    if response.status_code == 200:
                        return base64.b64encode(response.content).decode("utf-8")
                except Exception as e:
                    LOGGER.error(f"Error invoking flex checks http api: {e}")
                    raise ToolException(f"Failed to download image from discord. Status code: {response.status_code}")

            print("Breakpoint 2")
            discord_url_pattern = r"https?://media\.discordapp\.net/.*"
            is_discord_url = re.match(discord_url_pattern, image_path) is not None
            if is_discord_url and discord_token:
                image_base64 = fetch_image_from_discord(image_path)
                image_data_url = f"data:image/jpeg;base64,{image_base64}"
                content_block = {"type": "image_url", "image_url": {"url": image_data_url}}
            else:
                content_block = {"type": "image_url", "image_url": {"url": image_path}}

            # Call the Vision API
            response = client.chat.completions.create(
                # model="gpt-4-vision-preview",
                model=aiosettings.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}, content_block],
                    }
                ],
                max_tokens=900,
            )

            return response.choices[0].message.content
        except Exception as e:
            exc_type, exc_value, _ = sys.exc_info()
            err_msg = f"Error in read_image_tool: {e}"
            LOGGER.error(err_msg)
            raise ToolException(err_msg) from e

    async def _arun(
        self,
        image_path: str,
        prompt: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] | None = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("vision_tool does not support async")
