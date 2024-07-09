# Standard library imports
from __future__ import annotations

import asyncio
import json
import logging
import sys

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Type
from uuid import UUID

import langchain_chroma.vectorstores

from langchain import hub
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.pydantic_v1 import BaseModel, ConfigDict, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool
from langchain.tools.base import ToolException
from langchain_chroma import Chroma
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger as LOGGER

# from goob_ai.gen_ai.stores.paperstore import PaperStore
from goob_ai.llm_manager import LlmManager
from goob_ai.services.chroma_service import ChromaService


# from langchain_community.vectorstores import Chroma as ChromaVectorStore
# from goob_ai.clients.http_client import HttpClient
# from langchain.chains.retrieval_qa.base import RetrievalQA, VectorDBQA
# from langchain.chains import RetrievalQA
# from langchain.chains.retrieval_qa.base import RetrievalQA, VectorDBQA
# from langchain.tools import BaseTool as LangChainBaseTool
# from langchain.chains.summarize import load_summarize_chain
# from langchain.chat_models.base import BaseChatModel
# from langchain.docstore.document import Document
# from langchain.callbacks import HumanApprovalCallbackHandler
# from langchain.callbacks.base import BaseCallbackHandler
# from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
# https://github.com/Antony90/arxiv-discord/blob/9039612c5d346ab489e3c85e50b7f6f86a6348f4/ai/tools.py#L44


@dataclass
class PaperBackend:
    """
    Allows tools to refer to common objects.
    Specifically the chat_id to track mentioned papers in a chat. Is inserted into pre-prompt for better tool use
    """

    chat_id: str  # can track mentioned papers for a chat, for better tool use and easier prompting
    vectorstore: Chroma  # for getting, inserting, filtering, document embeddings
    # paper_store: PaperStore  # paper metadata: title, abstract, generated summaries
    llm: BaseLanguageModel  # for various Chains


# class BaseTool(LangChainBaseTool):
#     """Lets tools define a user friendly action text to be displayed in progress updates"""

#     action_label: str


# class BasePaperTool(BaseTool):
#     """Base class for tools which may want to load a paper before running their function."""

#     _backend: Optional[PaperBackend]
#     _text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

#     class Config:
#         model_config = ConfigDict(extra="allow")

#     # aliases to backend objects for subclasses
#     def llm(self):
#         return self._backend.llm

#     def paper_store(self):
#         return self._backend.paper_store

#     def vectorstore(self):
#         return self._backend.vectorstore

#     def set_backend(self, backend: PaperBackend):
#         self._backend = backend

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


# TOOL_ACTIONS = {}


# def register_tool_action(cls: BaseTool):
#     """A class decorator to track all tools, create a mapping which stores tool action labels"""
#     TOOL_ACTIONS[cls.name] = cls.action_label


class ReadTheDocsQASchema(BaseModel):
    user_question: str = Field(
        description="A question to ask about a readthedocs pdf. Cannot be empty. Must be a question abount opencv, rich, or Pillow."
    )
    # paper_id: str = Field(description="Substring of the Name of the paper to query")
    # paper_id: str = Field(description="ID of paper to query")


class ReadTheDocsQATool(BaseTool):
    # Must be unique within a set of tools provided to an LLM or agent.
    name = "chroma_question_answering"
    # Describes what the tool does. Used as context by the LLM or agent.
    # description = "Ask a question about the contents of a ReadTheDocs pdf for python modules opencv, rich, and Pillow. Primary source of factual information for a pdf. Don't include pdf ID/URL in the question."

    description = """You must use this tool for any questions or queries related to opencv, rich, and Pillow or substrings of it.
    This will return documents that are related to the user's question. The documents may not be always relevant to the user's question.
    If you use any of the documents returned to provide a helpful answer to user_question, please make sure to also return a valid URL of the document you used."""
    # Optional but recommended, can be used to provide more information (e.g., few-shot examples) or validation for expected parameters
    args_schema: Type[ReadTheDocsQASchema] = ReadTheDocsQASchema
    # Only relevant for agents. When True, after invoking the given tool, the agent will stop and return the result direcly to the user.
    return_direct: bool = False

    action_label = "Querying a paper"
    hub_prompt = hub.pull("rlm/rag-prompt")
    db: langchain_chroma.vectorstores.Chroma = Chroma(
        client=ChromaService.client,
        collection_name="readthedocs",
        embedding_function=OpenAIEmbeddings(),
    )
    model: ChatOpenAI | None = LlmManager().llm
    # Uses LLM for QA retrieval chain prompting
    # Vectorstore for embeddings of currently loaded PDFs

    def _run(self, user_question: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        # self.load_paper(paper_id)
        try:
            qa = self._make_qa_chain()
            answer = qa.run(user_question)
            LOGGER.debug(f"Answer: {answer}")
        except Exception as e:
            LOGGER.error(f"Error invoking flex checks http api: {e}")
            raise ToolException("Error invoking flex checks http api!")

        return answer

    async def _arun(self, user_question: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        # If the calculation is cheap, you can just delegate to the sync implementation
        # as shown below.
        # If the sync calculation is expensive, you should delete the entire _arun method.
        # LangChain will automatically provide a better implementation that will
        # kick off the task in a thread to make sure it doesn't block other async code.
        # await self.aload_paper(paper_id)
        qa = self._make_qa_chain()
        # await asyncio.sleep(0)  # placeholder for async code
        return qa.run(user_question, run_manager=run_manager.get_sync())

    def _make_qa_chain(self):
        """Make a RetrievalQA chain which filters by this paper_id"""
        # filter = {"source": paper_id}

        # retriever = self.vectorstore().as_retriever(search_kwargs={"filter": filter})
        # # TODO: generate multiple queries from different perspectives to pull a richer set of Documents

        # qa = RetrievalQA.from_chain_type(llm=self.llm(), chain_type="stuff", retriever=retriever)
        # Initialize the model with our deployment of Azure OpenAI
        # model = AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"])
        model = LlmManager().llm

        return RetrievalQA.from_chain_type(
            self.model,
            # The chain_type="stuff" lets LangChain take the list of matching documents from the retriever (Chroma DB in our case), insert everything all into a prompt, and pass it over to the llm.
            # SOURCE: https://www.gettingstarted.ai/tutorial-chroma-db-best-vector-database-for-langchain-store-embeddings/
            chain_type="stuff",
            retriever=self.db.as_retriever(),
            chain_type_kwargs={"prompt": self.hub_prompt},
        )
