# Standard library imports
from __future__ import annotations

import json
import logging
import sys

from dataclasses import dataclass
from typing import ClassVar, List, Optional, Type

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.tools.base import ToolException
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma as ChromaVectorStore
from loguru import logger as LOGGER

from goob_ai.clients.http_client import HttpClient
from goob_ai.gen_ai.stores.paperstore import PaperStore


@dataclass
class PaperBackend:
    """
    Allows tools to refer to common objects.
    Specifically the chat_id to track mentioned papers in a chat. Is inserted into pre-prompt for better tool use
    """

    chat_id: str  # can track mentioned papers for a chat, for better tool use and easier prompting
    vectorstore: Chroma  # for getting, inserting, filtering, document embeddings
    paper_store: PaperStore  # paper metadata: title, abstract, generated summaries
    llm: BaseLanguageModel  # for various Chains
