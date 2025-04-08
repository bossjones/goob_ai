"""goob_ai.tools.rag_tool"""

from __future__ import annotations

import logging
import sys
import traceback
from typing import Any, ClassVar, List, Optional, Type

import openai
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableMap,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_core.tools import BaseTool, ToolException
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from loguru import logger as LOGGER
from openai import Client

from goob_ai.aio_settings import aiosettings
from goob_ai.llm_manager import LlmManager

# TODO: Remove ChromaService import once reimplemented
# from goob_ai.services.chroma_service import ChromaService

RETRIEVAL_QA_CHAT_PROMPT: ChatPromptTemplate = hub.pull("langchain-ai/retrieval-qa-chat")
RAG_PROMPT: ChatPromptTemplate = hub.pull("rlm/rag-prompt")


def format_docs(docs: list[Document]) -> str:
    """Format a list of documents into a single string.

    Args:
        docs: List of Document objects to format.

    Returns:
        A string containing the concatenated page content of all documents.
    """
    return "\n\n".join(doc.page_content for doc in docs)


#####################################################################
# OUTPUT:
#####################################################################
# RETRIEVAL_QA_CHAT_PROMPT = {
#     'name': None,
#     'input_variables': ['context', 'input'],
#     'optional_variables': ['chat_history'],
#     'input_types': {
#         'chat_history': typing.List[typing.Union[langchain_core.messages.ai.AIMessage, langchain_core.messages.human.HumanMessage, langchain_core.messages.chat.ChatMessage,
# langchain_core.messages.system.SystemMessage, langchain_core.messages.function.FunctionMessage, langchain_core.messages.tool.ToolMessage]]
#     },
#     'output_parser': None,
#     'partial_variables': {'chat_history': []},
#     'metadata': {'lc_hub_owner': 'langchain-ai', 'lc_hub_repo': 'retrieval-qa-chat', 'lc_hub_commit_hash': 'b60afb6297176b022244feb83066e10ecadcda7b90423654c4a9d45e7a73cebc'},
#     'tags': None,
#     'messages': [
#         SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context'], template='Answer any use questions based solely on the context below:\n\n<context>\n{context}\n</context>')),
#         MessagesPlaceholder(variable_name='chat_history', optional=True),
#         HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}'))
#     ],
#     'validate_template': False
# }

# >>> rich.print(RAG_PROMPT.__dict__)
# {
#     'name': None,
#     'input_variables': ['context', 'question'],
#     'optional_variables': [],
#     'input_types': {},
#     'output_parser': None,
#     'partial_variables': {},
#     'metadata': {'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'},
#     'tags': None,
#     'messages': [
#         HumanMessagePromptTemplate(
#             prompt=PromptTemplate(
#                 input_variables=['context', 'question'],
#                 template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't
# know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
#             )
#         )
#     ],
#     'validate_template': False
# }
# >>>


# from langchain.chains.retrieval_qa.base import RetrievalQA

# from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain.base_language import BaseLanguageModel
# import asyncio
# import json
# import logging
# import sys

# from dataclasses import dataclass
# from goob_ai.gen_ai.stores.paperstore import PaperStore
# from langchain_community.vectorstores import Chroma as ChromaVectorStore
# from goob_ai.clients.http_client import HttpClient
# from langchain.chains.retrieval_qa.base import RetrievalQA, VectorDBQA
# from langchain.chains import RetrievalQA
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.retrieval_qa.base import RetrievalQA, VectorDBQA
# from langchain.tools import BaseTool as LangChainBaseTool
# from langchain.chains.summarize import load_summarize_chain
# from langchain.chat_models.base import BaseChatModel
# from langchain.docstore.document import Document
# from langchain.callbacks import HumanApprovalCallbackHandler
# from langchain.callbacks.base import BaseCallbackHandler
# from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
# https://github.com/Antony90/arxiv-discord/blob/9039612c5d346ab489e3c85e50b7f6f86a6348f4/ai/tools.py#L44


# @dataclass
# class PaperBackend:
#     """
#     Allows tools to refer to common objects.
#     Specifically the chat_id to track mentioned papers in a chat. Is inserted into pre-prompt for better tool use
#     """

#     chat_id: str  # can track mentioned papers for a chat, for better tool use and easier prompting
#     vectorstore: Chroma  # for getting, inserting, filtering, document embeddings
#     # paper_store: PaperStore  # paper metadata: title, abstract, generated summaries
#     llm: BaseLanguageModel  # for various Chains


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


# Add typing for input
class Question(BaseModel):
    __root__: str


# TODO: Reimplement BaseChromaDBTool without langchain_chroma dependency
class BaseChromaDBTool(BaseModel):
    """Base tool for interacting with vector store."""

    # TODO: Update field type once Chroma dependency is removed
    db: Any = Field(
        exclude=True,
        title="db",
        description="vector store client for getting, inserting, filtering, document embeddings.",
    )

    hub_prompt: ChatPromptTemplate = RAG_PROMPT
    llm: ChatOpenAI = Field(exclude=True, title="llm", description="Large Language model to use for embedding.")

    class Config(BaseTool.Config):
        """Pydantic config."""

        arbitrary_types_allowed = True


class ReadTheDocsQASchema(BaseModel):
    """Schema for ReadTheDocs QA tool input.

    You must use this tool for any questions or queries related to opencv, rich, and Pillow or substrings of it.
    This will return documents that are related to the user's question. The documents may not be always relevant
    to the user's question. If you use any of the documents returned to provide a helpful answer to question,
    please make sure to also return a valid URL of the document you used.

    Args:
        question: A question to ask about a readthedocs pdf. Cannot be empty. Must be a question about opencv, rich, or Pillow.
    """

    question: str = Field(
        description="The question to ask about the documentation.",
    )


# TODO: Reimplement ReadTheDocsQATool without Chroma dependency
class ReadTheDocsQATool(BaseChromaDBTool, BaseTool):
    """Tool for answering questions about ReadTheDocs documentation."""

    name: str = "chroma_question_answering"
    description: str = "You must use this tool for any questions or queries related to opencv, rich, and Pillow or substrings of it. This will return documents that are related to the user's question. The documents may not be always relevant to the user's question. If you use any of the documents returned to provide a helpful answer to question, please make sure to also return a valid URL of the document you used."
    args_schema: type[ReadTheDocsQASchema] = ReadTheDocsQASchema
    return_direct: bool = False
    handle_tool_error: bool = False

    def _run(self, question: str, **kwargs) -> str:
        """Run the tool synchronously.

        Args:
            question: The question to answer.
            **kwargs: Additional keyword arguments.

        Returns:
            The answer to the question.

        Raises:
            NotImplementedError: This method is currently disabled.
        """
        # TODO: Reimplement _run without Chroma dependency
        raise NotImplementedError("ReadTheDocsQATool._run is currently disabled")

    async def _arun(self, question: str, **kwargs) -> str:
        """Run the tool asynchronously.

        Args:
            question: The question to answer.
            **kwargs: Additional keyword arguments.

        Returns:
            The answer to the question.

        Raises:
            NotImplementedError: This method is currently disabled.
        """
        # TODO: Reimplement _arun without Chroma dependency
        raise NotImplementedError("ReadTheDocsQATool._arun is currently disabled")

    @traceable
    def _get_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        """Get the retriever for the tool.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            A VectorStoreRetriever instance.

        Raises:
            NotImplementedError: This method is currently disabled.
        """
        # TODO: Reimplement _get_retriever without Chroma dependency
        raise NotImplementedError("ReadTheDocsQATool._get_retriever is currently disabled")

    @traceable
    def _make_qa_chain(self) -> RunnableSerializable[Any, str]:
        """Create the QA chain for the tool.

        Returns:
            A RunnableSerializable instance.

        Raises:
            NotImplementedError: This method is currently disabled.
        """
        # TODO: Reimplement _make_qa_chain without Chroma dependency
        raise NotImplementedError("ReadTheDocsQATool._make_qa_chain is currently disabled")
