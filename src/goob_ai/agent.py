# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
# pylint: disable=no-name-in-module

from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import openai

from boto3.session import Session as boto3_Session
from langchain.agents import AgentExecutor
from langchain.agents.agent import BaseMultiActionAgent, BaseSingleActionAgent
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.callbacks.tracers import LoggingCallbackHandler
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.globals import set_debug
from langchain.pydantic_v1 import BaseModel, Field
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableField, Runnable, RunnableBranch, RunnableLambda, RunnableMap
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from loguru import logger as LOGGER
from openai import Client
from pydantic_settings import SettingsConfigDict

from goob_ai import llm_manager, redis_memory
from goob_ai.aio_settings import AioSettings, aiosettings
from goob_ai.gen_ai.tools.vision_tool import VisionTool
from goob_ai.llm_manager import LlmManager
from goob_ai.services.chroma_service import ChromaService
from goob_ai.tools.rag_tool import ReadTheDocsQATool


if TYPE_CHECKING:
    from pinecone.control import Pinecone  # pyright: ignore[reportAttributeAccessIssue]


# class AiAgent(BaseModel):
class AiAgent:
    custom_tools: list[BaseTool] | None = None
    all_tools: list[BaseTool] | None = None
    agent: Union[BaseSingleActionAgent, BaseMultiActionAgent] | None = None
    settings: AioSettings | None = None
    dynamodb_session: boto3_Session | None = None
    agent_name: str | None = None
    agent_created_by: str | None = None
    agent_purpose: str | None = None
    agent_personality: str | None = None
    logging_handler: LoggingCallbackHandler | None = None
    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    def __init__(self):
        super().__init__()
        # Load the settings from config files and environment variables:
        self.settings = aiosettings
        # initialize stuff:
        self.init_agent_name()
        self.init_tools()
        self.init_agent_executor()
        # global LangChain debugging:
        if self.settings.langchain_debug_logs:
            set_debug(True)

        self._vector_store: Optional[Chroma] = None
        self._embeddings: Optional[OpenAIEmbeddings] = None

    @property
    def embeddings(self) -> Chroma:
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings()
            LOGGER.debug(f"Setting default embeddings: {self._embeddings}")
        return self._embeddings

    @property
    def vector_store(self) -> Chroma:
        if self._vector_store is None:
            self._vector_store = Chroma(
                client=ChromaService.client,
                collection_name="readthedocs",
                embedding_function=self._embeddings,
            )
            LOGGER.debug(f"Setting default vector store: {self._vector_store}")
        return self._vector_store

    # FIXME: Implement meme personality as well. https://chatgptaihub.com/chatgpt-prompts-for-memes/
    def init_agent_name(self):
        # Initialize the agent name, purpose, created by and personality
        self.agent_name = "Malcolm Jones Developer Assitant 'GOOBS' as an homage to his frenchbulldog Gaston aka GOOBS"
        self.agent_created_by = "Tony Dark himself, Malcolm Jones"
        self.agent_purpose = "help our developers build better software faster"
        self.agent_personality = "You have a geeky and clever sense of humor"

    @traceable
    def get_prompt(self) -> ChatPromptTemplate:
        # write a docstring using pep257 conventions
        """Define the chat prompt for the agent."""
        # We wouldn't typically know what the users prompt is beforehand, so we actually want to add this in. So rather than writing the prompt directly, we create a `PromptTemplate` with a single input variable `query`.
        # Define the chat prompt

        # NOTE: Chat message types:
        # ***************************************************
        # HumanMessage: A message sent from the perspective of the human
        # AIMessage: A message sent from the perspective of the AI the human is interacting with
        # SystemMessage: A message setting the objectives the AI should follow
        # ChatMessage: A message allowing for arbitrary setting of role. You won't be using this too much
        # SOURCE: https://github.com/gkamradt/langchain-tutorials/blob/main/chatapi/ChatAPI%20%2B%20LangChain%20Basics.ipynb
        # ***************************************************
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
             You are a helpful AI assistant called {self.agent_name}.
             Use the following pieces of context to answer the question at the end.
             You were created by {self.agent_created_by} to {self.agent_purpose}.
             You think step by step about the user request and provide a helpful and truthful response.

             Very Important: If the question is about writing code use backticks (```) at the front and end of the code snippet and include the language use after the first ticks.

             If the user provides an image use Custom Tool vision_api to get more information about the image then pass the text along with the original question to vector_store_tool.
             You remember {self.settings.chat_history_buffer} previous messages from the chat thread.
             If you use documents from any tools to provide a helpful answer to user question, please make sure to also return a valid URL of the document you used
             If you don't know the answer, just say so, and don't try to make anything up. DO NOT allow made up or fake answers.
             If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
             Use as much detail when as possible when responding.
             All answers should be in MARKDOWN (.md) Format:
             {self.agent_personality}.
             """,
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

    def init_tools(self):
        self.custom_tools: Union[list[BaseTool], list[Any]] | None = [VisionTool()]
        self._embeddings = embeddings = OpenAIEmbeddings()
        self._vector_store = db = Chroma(
            client=ChromaService.client,
            collection_name="readthedocs",
            embedding_function=self._embeddings,
        )
        llm = llm_manager.LlmManager().llm
        rtd_tool = ReadTheDocsQATool(db=db, llm=llm)
        self.custom_tools.append(rtd_tool)

        # self.custom_tools: Union[list[BaseTool], list[Any]] | None = [VisionTool()]
        # ***************************************************
        # NOTE: CustomTool Error handling
        # ***************************************************
        # See: https://python.langchain.com/docs/modules/tools/custom_tools/#handling-tool-error
        # To ensure continuous execution when a tool encounters an error, raise a ToolException and set handle_tool_error.
        # Here is why: https://github.com/langchain-ai/langchain/issues/7715
        for t in self.custom_tools:
            t.handle_tool_error = True

        self.all_tools = self.custom_tools

    # # SOURCE: https://github.com/Haste171/langchain-chatbot/blob/main/handlers/base.py
    # @traceable
    def init_agent_executor(self) -> None:
        """Initalize agent executor."""
        llm_with_tools = LlmManager().llm.bind_tools(tools=[convert_to_openai_tool(t) for t in self.all_tools])

        self.prompt = self.get_prompt()

        self.logging_handler = LoggingCallbackHandler(logger=LOGGER)

        # Define the agent
        self.agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
                "chat_history": lambda x: x["chat_history"],
            }
            | self.prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )  # pyright: ignore[reportAttributeAccessIssue]

    # def setup_agent_executor(self, session_id: str, user_task: str):
    @traceable
    def setup_agent_executor(self, session_id: str, user_task: str) -> AgentExecutor:
        LOGGER.debug(f"session_id = {session_id}")
        LOGGER.debug(f"user_task = {user_task}")
        # ttl_in_seconds = self.settings.dynamodb_ttl_days * 24 * 60 * 60
        # FIXME: replace foo with a proper session_id later

        if aiosettings.experimental_redis_memory:
            LOGGER.error("Using experimental redis memory")
            memory = redis_memory.create_memory_instance(session_id=session_id, store_ai_answer=True)
        else:
            LOGGER.error("Using DEFAULT redis memory")
            message_history = RedisChatMessageHistory("foo", url=f"{aiosettings.redis_url}", key_prefix="goob:")

            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                chat_memory=message_history,
                return_messages=True,
                k=self.settings.chat_history_buffer,
                output_key="output",
            )

        return AgentExecutor(
            agent=self.agent,
            tools=self.all_tools,
            verbose=True,
            callbacks=[self.logging_handler, StdOutCallbackHandler()],
            memory=memory,
        )

    @traceable
    def process_user_task(self, session_id: str, user_task: str) -> dict[str, Any]:
        """
        Summary:
        Process a user task by invoking an agent executor and returning the output.

        Explanation:
        This function processes a user task by setting up an agent executor with the provided session ID and user task. It then invokes the agent executor with the user task input and returns the output generated by the agent. If an error occurs during processing, it logs the exception and returns an error message.

        Args:
        ----
        - self: The instance of the class.
        - session_id (str): The session ID for the user task.
        - user_task (str): The user task to be processed.

        Returns:
        -------
        - str: The output generated by the agent or an error message if processing fails.

        """
        LOGGER.debug(f"session_id = {session_id}")
        LOGGER.debug(f"user_task = {user_task}")

        try:
            agent_executor = self.setup_agent_executor(session_id, user_task)
            config = {"metadata": {"session_id": session_id}}
            result = agent_executor.invoke({"input": user_task}, config=config)
            LOGGER.error(f"type(result) = {type(result)}")
            return result.get("output", "No response generated by the agent.")
        except Exception as e:
            LOGGER.exception(f"Error in process_user_task: {e}")
            return "An error occurred while processing the task."

    @traceable
    async def process_user_task_streaming(self, session_id: str, user_task: str):
        try:
            agent_executor = self.setup_agent_executor(session_id, user_task)
            # Loop through the agent executor stream
            async for chunk in agent_executor.astream_log({"input": user_task}):
                # We want to pull any tool invocation and the tokens for the final response
                for op in chunk.ops:
                    # Extract the tool invocation from the agent executor stream
                    if op["op"] == "add" and "/logs/OpenAIToolsAgentOutputParser/final_output" in op["path"]:
                        try:
                            tool_invocations = op["value"]["output"]
                            for tool_invocation in tool_invocations:
                                # Directly access the 'log' attribute of the tool_invocation object
                                if hasattr(tool_invocation, "log"):
                                    LOGGER.info(f"Tool invocation: {tool_invocation.log}")
                                    yield tool_invocation.log

                        except AttributeError as e:
                            LOGGER.exception(f"An error occurred: {e}")
                            continue

                    # Logic to pull agent response from the streaming output
                    # Target paths ending with 'streamed_output_str/-', and check if 'value' is non-empty
                    if op["op"] == "add" and op["path"].endswith("/streamed_output_str/-") and op["value"]:
                        value = op["value"]  # Directly access the value
                        LOGGER.info(f"Chunk: {value}")
                        yield value

        except Exception as e:
            LOGGER.exception(f"Error in process_user_task_streaming: {e}")
            yield "An error occurred while processing the task."

    @traceable
    def summarize(self, user_input: str) -> str:
        """
        Process the user input and summarize it using a LLM. This method is used for the summarization API.

        :param user_input: The task input by the user.
        :return: The output from the LLM.
        """
        try:
            # Setup a LLM instance
            llm = LlmManager().llm

            # Setup the prompt
            prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Please summarize the input from the user. Just provide the text for the summary, please don't add any additional information or commentary.",
                    ),
                    ("user", "{user_input}"),
                ]
            )

            # Put this in a chain
            chain = prompt | llm | StrOutputParser()
            chain_custom_name = chain.with_config({"run_name": "summarize"})
            return chain_custom_name.invoke({"user_input": user_input})

        except Exception as e:
            LOGGER.exception(f"Error during summarization of user task: {e}")
            return "An error occurred while summarizing the input."
