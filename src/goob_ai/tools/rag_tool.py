"""goob_ai.tools.rag_tool"""

from __future__ import annotations

import logging
import sys
import traceback

from typing import ClassVar, List, Optional, Type

import langchain_chroma.vectorstores
import openai

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import Document
from langchain.schema.runnable import ConfigurableField, Runnable, RunnableBranch, RunnableLambda, RunnableMap
from langchain.tools import BaseTool
from langchain.tools.base import ToolException
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough, RunnableSerializable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from loguru import logger as LOGGER
from openai import Client

from goob_ai.aio_settings import aiosettings
from goob_ai.llm_manager import LlmManager
from goob_ai.services.chroma_service import ChromaService


# from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
# from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
RETRIEVAL_QA_CHAT_PROMPT: ChatPromptTemplate = hub.pull("langchain-ai/retrieval-qa-chat")
RAG_PROMPT: ChatPromptTemplate = hub.pull("rlm/rag-prompt")


def format_docs(docs: List[Document]):
    """_summary_

    Args:
        docs (List[Document]): _description_

    Returns:
        _type_: _description_
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


class BaseChromaDBTool(BaseModel):
    """Base tool for interacting with Chroma."""

    # db: SQLDatabase = Field(exclude=True)
    db: langchain_chroma.vectorstores.Chroma = Field(exclude=True)

    hub_prompt = ChatPromptTemplate = RAG_PROMPT
    # db: langchain_chroma.vectorstores.Chroma = Field(
    #     default_factory=lambda: Chroma(
    #         client=ChromaService.client,
    #         collection_name="readthedocs",
    #         embedding_function=OpenAIEmbeddings(),
    # )
    llm: ChatOpenAI = Field(exclude=True)
    # model: ClassVar[ChatOpenAI] | None = LlmManager().llm
    # llm_chain: LLMChain = Field(
    #     default_factory=lambda: LLMChain(
    #         llm=OpenAI(temperature=0),
    #         prompt=PromptTemplate(
    #             template=QUERY_CHECKER, input_variables=["query", "dialect"]
    #         ),
    #     )
    # )

    class Config(BaseTool.Config):
        pass


class ReadTheDocsQASchema(BaseModel):
    question: str = Field(
        description="A question to ask about a readthedocs pdf. Cannot be empty. Must be a question abount opencv, rich, or Pillow."
    )
    # paper_id: str = Field(description="Substring of the Name of the paper to query")
    # paper_id: str = Field(description="ID of paper to query")


class ReadTheDocsQATool(BaseChromaDBTool, BaseTool):
    # Must be unique within a set of tools provided to an LLM or agent.
    name: str = "chroma_question_answering"
    # Describes what the tool does. Used as context by the LLM or agent.
    # description = "Ask a question about the contents of a ReadTheDocs pdf for python modules opencv, rich, and Pillow. Primary source of factual information for a pdf. Don't include pdf ID/URL in the question."

    description: str = "You must use this tool for any questions or queries related to opencv, rich, and Pillow or substrings of it. This will return documents that are related to the user's question. The documents may not be always relevant to the user's question. If you use any of the documents returned to provide a helpful answer to question, please make sure to also return a valid URL of the document you used."
    # Optional but recommended, can be used to provide more information (e.g., few-shot examples) or validation for expected parameters
    args_schema: Type[ReadTheDocsQASchema] = ReadTheDocsQASchema
    # Only relevant for agents. When True, after invoking the given tool, the agent will stop and return the result direcly to the user.
    return_direct: bool = False
    handle_tool_error: bool = False

    # action_label: str = "Querying a paper"
    # hub_prompt = ClassVar[ChatPromptTemplate] = hub.pull("rlm/rag-prompt")
    # hub_prompt = ChatPromptTemplate = RAG_PROMPT
    # db: ClassVar[langchain_chroma.vectorstores.Chroma] = Chroma(
    #     client=ChromaService.client,
    #     collection_name="readthedocs",
    #     embedding_function=OpenAIEmbeddings(),
    # )
    # model: ClassVar[ChatOpenAI] | None = LlmManager().llm
    # llm_chain: LLMChain = Field(
    #     default_factory=lambda: LLMChain(
    #         llm=OpenAI(temperature=0),
    #         prompt=PromptTemplate(
    #             template=QUERY_CHECKER, input_variables=["query", "dialect"]
    #         ),
    #     )
    # )
    # Uses LLM for QA retrieval chain prompting
    # Vectorstore for embeddings of currently loaded PDFs

    # def _run(self, question: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
    def _run(self, question: str) -> str:
        """Use the tool."""
        # self.load_paper(paper_id)
        # import bpdb

        # bpdb.set_trace()
        try:
            qa = self._make_qa_chain()
            answer = qa.invoke(question)
            # answer = qa.invoke({"input": question})
            # answer = qa.invoke({"question": question})
            # answer = qa.run(question)
            LOGGER.debug(f"Answer: {answer}")
        except Exception as e:
            # print(f"{e}")
            # exc_type, exc_value, exc_traceback = sys.exc_info()
            # print(f"Error Class: {e.__class__}")
            # output = f"[UNEXPECTED] {type(e).__name__}: {e}"
            # print(output)
            # print(f"exc_type: {exc_type}")
            # print(f"exc_value: {exc_value}")
            # traceback.print_tb(exc_traceback)
            # bpdb.pm()
            LOGGER.error(f"Error invoking {self.name}: {e}")
            raise ToolException(f"Error invoking {self.name}!") from e

        return answer

    # async def _arun(self, question: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
    async def _arun(self, question: str) -> str:
        """Use the tool asynchronously."""
        # If the calculation is cheap, you can just delegate to the sync implementation
        # as shown below.
        # If the sync calculation is expensive, you should delete the entire _arun method.
        # LangChain will automatically provide a better implementation that will
        # kick off the task in a thread to make sure it doesn't block other async code.
        # await self.aload_paper(paper_id)
        qa = self._make_qa_chain()
        answer = qa.invoke(question)
        # return qa.invoke(question, run_manager=run_manager.get_sync())
        return answer

    # def _setup(self):
    #     self.hub_prompt = RAG_PROMPT
    #     self.db: langchain_chroma.vectorstores.Chroma = Chroma(
    #         client=ChromaService.client,
    #         collection_name="readthedocs",
    #         embedding_function=OpenAIEmbeddings(),
    #     )
    #     self.model: ChatOpenAI | None = LlmManager().llm

    @traceable
    def _make_qa_chain(self):
        """Make a RetrievalQA chain which filters by this paper_id"""
        # filter = {"source": paper_id}

        # retriever = self.vectorstore().as_retriever(search_kwargs={"filter": filter})
        # # TODO: generate multiple queries from different perspectives to pull a richer set of Documents

        # qa = RetrievalQA.from_chain_type(llm=self.llm(), chain_type="stuff", retriever=retriever)
        # Initialize the model with our deployment of Azure OpenAI
        # model = AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"])
        # model = LlmManager().llm

        retriever = self.db.as_retriever()
        # NOTE: This looks like the future but we're going to use the old school way
        ####################################################################################
        # combine_docs_chain = create_stuff_documents_chain(self.llm, RAG_PROMPT)
        # retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        ####################################################################################

        # RAG chain
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

        # >>> chain
        # ReadTheDocsQATool(db=<langchain_chroma.vectorstores.Chroma object at 0x1713777f0>, llm=ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x17154ab30>, async_client=<openai.reso
        # urces.chat.completions.AsyncCompletions object at 0x171564250>, model_name='gpt-4o-2024-05-13', temperature=0.1, openai_api_key=SecretStr('**********'), openai_proxy='', streaming=True))
        # >>> qa = rtd_tool._make_qa_chain()
        # >>> qa
        # {
        #   context: VectorStoreRetriever(tags=['Chroma', 'OpenAIEmbeddings'], vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x1713777f0>),
        #   question: RunnablePassthrough()
        # }
        # | ChatPromptTemplate(input_variables=['context', 'question'], metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c4
        # 36e6cdcb6e'}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces o
        # f retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {contex
        # t} \nAnswer:"))])
        # | ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x17154ab30>, async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x171564250>, model_name='gpt-4o-20
        # 24-05-13', temperature=0.1, openai_api_key=SecretStr('**********'), openai_proxy='', streaming=True)
        # | StrOutputParser()
        # >>> type(qa)
        # <class 'langchain_core.runnables.base.RunnableSequence'>
        # >>>

        # import bpdb
        # bpdb.set_trace()

        # question_answer_chain = create_stuff_documents_chain(
        #     self.model,
        #     # The chain_type="stuff" lets LangChain take the list of matching documents from the retriever (Chroma DB in our case), insert everything all into a prompt, and pass it over to the llm.
        #     # SOURCE: https://www.gettingstarted.ai/tutorial-chroma-db-best-vector-database-for-langchain-store-embeddings/
        #     chain_type="stuff",
        #     retriever=retriever,
        #     chain_type_kwargs={"prompt": self.hub_prompt},
        # )
        # question_answer_chain = create_stuff_documents_chain(self.llm, self.hub_prompt)
        # return create_retrieval_chain(retriever, question_answer_chain)
        return chain
