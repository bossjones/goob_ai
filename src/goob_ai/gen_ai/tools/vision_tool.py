"""goob_ai.gen_ai.tools.vision_tool: This module contains the VisionTool class for the Goob AI application."""

# pylint: disable=no-member
# mypy: disable-error-code="return"
# mypy: disable-error-code="str-byte-safe"
# mypy: disable-error-code="misc"
from __future__ import annotations

import base64
import logging
import re
import sys
import uuid

from typing import ClassVar, Dict, Optional, Type

import requests

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.tools import ToolException
from langchain_openai import ChatOpenAI
from loguru import logger as LOGGER
from openai import Client

from goob_ai.aio_settings import aiosettings
from goob_ai.clients.http_client import HttpClient
from goob_ai.llm_manager import VisionModel


DISCORD_URL_PATTERN = r"https?://media\.discordapp\.net/.*"


def get_pattern() -> str:
    return DISCORD_URL_PATTERN


# SOURCE: https://github.com/BiscuitBobby/orchestrator/blob/11e9bc08ea80b2581a50383554f9669131ed13d3/Functions/discord_message.py#L46
# def download_discord_images_via_api(url):
#     url = 'https://discord.com/api/v9/users/@me/channels'
#     headers = {
#         'Authorization': f'Bot {bot_token}',
#         'Content-Type': 'application/json',
#     }
#     payload = {
#         'recipient_id': user_id
#     }
#     response = requests.post(url, headers=headers, json=payload)
#     if response.status_code == 200:
#         # Successfully retrieved the DM channel
#         return response.json()['id']
#     else:
#         # Handle errors
#         print(f"Error getting DM channel: {response.status_code}")
#         print(response.json())
#         return None


class VisionToolInput(BaseModel):
    image_path: str = Field(description="The URL to the image file.")
    prompt: str = Field(description="The prompt to use for the API call.")


class VisionTool(BaseTool):
    name: str = "vision_api"
    description: str = (
        "This tool calls OpenAI's Vision API to get more information about an image given a URL to an image file."
    )
    args_schema: Type[BaseModel] = VisionToolInput
    return_direct: bool = False
    handle_tool_error: bool = True

    def _run(self, image_path: str, prompt: str, **kwargs) -> str | bytes:
        """
        Use this tool to get more information about an image given a URL to an image file. Use for all urls for image files including discord urls.

        Args:
            image_path: The URL to the image file
            prompt: The prompt to use for the API call

        Returns: The response from the Vision API
        """

        LOGGER.info(f"image_path = {image_path}")
        LOGGER.info(f"prompt = {prompt}")
        try:
            # Initialize the Vision API client
            client: Client | None = VisionModel().vision_api
            # client = Client(aiosettings.openai_api_key.get_secret_value())
            # API_BASE_URL: https://api.groq.com/openai/v1/

            # Function to download image from discord and convert to base64
            def fetch_image_from_discord(url: str) -> str | bytes | None:
                # Initialize the discord settings
                discord_token = aiosettings.discord_token.get_secret_value()
                headers = {
                    "Authorization": f"Bot {discord_token.get_secret_value()}",
                    "Content-Type": "application/json",
                }
                # Check if the message content is a URL
                url_pattern = re.compile(
                    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
                )

                try:
                    # response: requests.Response = requests.get(url)
                    response: requests.Response = requests.get(url, headers=headers)
                    if response.status_code == 200:
                        return base64.b64encode(response.content).decode("utf-8")
                except Exception as e:
                    LOGGER.error(f"Error invoking flex checks http api: {e}")
                    raise ToolException(f"Failed to download image from discord. Status code: {response.status_code}")

            print("Breakpoint 2")
            discord_url_pattern = get_pattern()
            LOGGER.debug(f"discord_url_pattern = {discord_url_pattern}")
            is_discord_url = re.match(discord_url_pattern, image_path) is not None
            if is_discord_url and discord_token:
                LOGGER.info("Found an image in query!")
                image_base64 = fetch_image_from_discord(image_path)
                image_data_url = f"data:image/jpeg;base64,{image_base64}"
                content_block = {"type": "image_url", "image_url": {"url": image_data_url}}
            else:
                content_block = {"type": "image_url", "image_url": {"url": image_path}}

            # import bpdb
            # bpdb.set_trace()
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
            )  # pyright: ignore[reportAttributeAccessIssue]

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

    # async def get_image()
