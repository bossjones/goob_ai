# Standard library imports
from __future__ import annotations

import json
import logging
import sys

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Type
from uuid import UUID

from langchain import LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.callbacks import HumanApprovalCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models.base import BaseChatModel
from langchain.docstore.document import Document
from langchain.pydantic_v1 import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool as LangChainBaseTool
from langchain.tools.base import ToolException
from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma as ChromaVectorStore
from loguru import logger as LOGGER
from pydantic import BaseModel, Extra, Field

from goob_ai.clients.http_client import HttpClient
from goob_ai.gen_ai.stores.paperstore import PaperStore


# https://github.com/Antony90/arxiv-discord/blob/9039612c5d346ab489e3c85e50b7f6f86a6348f4/ai/tools.py#L44


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


class BaseTool(LangChainBaseTool):
    """Lets tools define a user friendly action text to be displayed in progress updates"""

    action_label: str


class BasePaperTool(BaseTool):
    """Base class for tools which may want to load a paper before running their function."""

    _backend: Optional[PaperBackend]
    _text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    class Config:
        extra = Extra.allow

    # aliases to backend objects for subclasses
    def llm(self):
        return self._backend.llm

    def paper_store(self):
        return self._backend.paper_store

    def vectorstore(self):
        return self._backend.vectorstore

    def set_backend(self, backend: PaperBackend):
        self._backend = backend

    # def load_paper(self, paper_id: str) -> bool:
    #     """Load a paper. Will download if it doesn't exist in vectorstore.
    #     return: Whether it was already in the vectorstore."""
    #     if self._backend is None:
    #         raise Exception(f"No paper backend to load paper `{paper_id}`")

    #     # check for existing Docs of this paper
    #     result = self._backend.vectorstore.get(where={"source":paper_id})
    #     if len(result["documents"]) != 0: # any key can be checked
    #         found = True # already in db
    #     else:
    #         doc, abstract = arxiv_fetch.get_doc_sync(paper_id)
    #         self._backend.paper_store.save_title_abstract(paper_id, doc.metadata["title"], abstract)

    #         # split and embed docs in vectorstore
    #         split_docs = self._text_splitter.split_documents([doc])
    #         self._backend.vectorstore.add_documents(split_docs)
    #         found = False

    #     self._backend.paper_store.add_mentioned_paper(paper_id, self._backend.chat_id)
    #     return found

    # async def aload_paper(self, paper_id: str) -> bool:
    #     """Load a paper. Will download if it doesn't exist in vectorstore.
    #     return: Whether it was already in the vectorstore."""
    #     if self._backend is None:
    #         raise Exception(f"No paper backend to load paper `{paper_id}`")

    #     # check for existing Docs of this paper
    #     result = self._backend.vectorstore.get(where={"source":paper_id})
    #     if len(result["documents"]) != 0: # any key can be checked
    #         found = True # already in db
    #     else:
    #         doc, abstract = await arxiv_fetch.get_doc_async(paper_id)
    #         self._backend.paper_store.save_title_abstract(paper_id, doc.metadata["title"], abstract)

    #         # split and embed docs in vectorstore
    #         split_docs = self._text_splitter.split_documents([doc])
    #         self._backend.vectorstore.add_documents(split_docs) # TODO: find store with async implementation
    #         found = False

    #     self._backend.paper_store.add_mentioned_paper(paper_id, self._backend.chat_id)
    #     return found


TOOL_ACTIONS = {}


def register_tool_action(cls: BaseTool):
    """A class decorator to track all tools, create a mapping which stores tool action labels"""
    TOOL_ACTIONS[cls.name] = cls.action_label


class ReadTheDocsQASchema(BaseModel):
    question: str = Field(
        description="A question to ask about a readthedocs pdf. Cannot be empty. Do not include the paper ID"
    )
    paper_id: str = Field(description="ID of paper to query")


class ReadTheDocsQATool(BasePaperTool):
    name = "chroma_question_answering"
    description = "Ask a question about the contents of a pdf. Primary source of factual information for a pdf. Don't include pdf ID/URL in the question."
    args_schema: Type[ReadTheDocsQASchema] = ReadTheDocsQASchema

    action_label = "Querying a paper"
    # Uses LLM for QA retrieval chain prompting
    # Vectorstore for embeddings of currently loaded PDFs

    def _run(self, question, paper_id) -> str:
        self.load_paper(paper_id)
        qa = self._make_qa_chain(paper_id)
        return qa.run(question)

    async def _arun(self, question, paper_id) -> str:
        await self.aload_paper(paper_id)
        qa = self._make_qa_chain(paper_id)
        return qa.run(question)

    def _make_qa_chain(self, paper_id: str):
        """Make a RetrievalQA chain which filters by this paper_id"""
        filter = {"source": paper_id}

        retriever = self.vectorstore().as_retriever(search_kwargs={"filter": filter})
        # TODO: generate multiple queries from different perspectives to pull a richer set of Documents

        qa = RetrievalQA.from_chain_type(llm=self.llm(), chain_type="stuff", retriever=retriever)
        return qa
