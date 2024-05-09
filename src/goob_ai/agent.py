# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

import logging

from typing import Any, List, Union

from boto3.session import Session as boto3_Session
from langchain.agents import AgentExecutor
from langchain.agents.agent import BaseMultiActionAgent, BaseSingleActionAgent
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.callbacks.tracers import LoggingCallbackHandler
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.globals import set_debug
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from loguru import logger as LOGGER
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict

from goob_ai.aio_settings import AioSettings, aiosettings
from goob_ai.gen_ai.tools.vision_tool import VisionTool
from goob_ai.llm_manager import LlmManager


class AiAgent(BaseModel):
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

    # FIXME: Implement meme personality as well. https://chatgptaihub.com/chatgpt-prompts-for-memes/
    def init_agent_name(self):
        # Initialize the agent name, purpose, created by and personality
        self.agent_name = "Malcolm Jones Developer Assitant 'GOOBS' as an homage to his frenchbulldog Gaston aka GOOBS"
        self.agent_created_by = "Tony Dark himself, Malcolm Jones"
        self.agent_purpose = "help our developers build better software faster"
        self.agent_personality = "You have a geeky and clever sense of humor"

    def init_tools(self):
        self.custom_tools: list[BaseTool] | None = [VisionTool()]
        # ***************************************************
        # NOTE: CustomTool Error handling
        # ***************************************************
        # See: https://python.langchain.com/docs/modules/tools/custom_tools/#handling-tool-error
        # To ensure continuous execution when a tool encounters an error, raise a ToolException and set handle_tool_error.
        # Here is why: https://github.com/langchain-ai/langchain/issues/7715
        for t in self.custom_tools:
            t.handle_tool_error = True

        self.all_tools = self.custom_tools

    # def init_dynamodb_session_and_table(self):
    #     """
    #     Initializes the dynamodb session and validates that the table exists
    #     """
    #     self.dynamodb_session = boto3_Session(region_name=self.settings.aws_region)

    #     dynamodb_client = self.dynamodb_session.client("dynamodb", endpoint_url=self.settings.dynamodb_endpoint_url)
    #     LOGGER.info(
    #         f"Set up dynamodb client with endpoint: {self.settings.dynamodb_endpoint_url}, aws region: {self.settings.aws_region}"
    #     )

    #     try:
    #         response = dynamodb_client.describe_table(TableName=self.settings.dynamodb_table_name)
    #         if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
    #             LOGGER.info("DynamoDB table exists")
    #         else:
    #             LOGGER.error(
    #                 f"DynamoDB table 'describe' action failed with code {response['ResponseMetadata']['HTTPStatusCode']}"
    #             )
    #             raise
    #     except dynamodb_client.exceptions.ResourceNotFoundException:
    #         if self.settings.env_name == LOCAL_ENV_NAME:
    #             LOGGER.info("DynamoDB table doesn't exist and environment is local. Creating the table.")
    #             self.create_dynamodb_table(dynamodb_client)
    #             pass
    #         else:
    #             LOGGER.error("DynamoDB table doesn't exist and environment is not local. Exiting.")
    #             raise

    # def create_dynamodb_table(self, dynamodb_client):
    #     """
    #     Creates the DynamoDB table; should be called only for local environment
    #     :param dynamodb_client:
    #     :return:
    #     """
    #     try:
    #         response = dynamodb_client.create_table(
    #             AttributeDefinitions=[
    #                 {
    #                     "AttributeName": "SessionId",
    #                     "AttributeType": "S",
    #                 },
    #             ],
    #             KeySchema=[
    #                 {
    #                     "AttributeName": "SessionId",
    #                     "KeyType": "HASH",
    #                 },
    #             ],
    #             # BillingMode doesn't matter, this is done only for local env
    #             BillingMode="PAY_PER_REQUEST",
    #             TableName=self.settings.dynamodb_table_name,
    #         )
    #         if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
    #             LOGGER.info("DynamoDB table created")
    #         else:
    #             LOGGER.error("DynamoDB table creation failed ")
    #             raise
    #     except dynamodb_client.exceptions.ResourceInUseException:
    #         LOGGER.info("DynamoDB table already exists")
    #         pass

    def init_agent_executor(self):
        llm_with_tools = LlmManager().llm.bind_tools(tools=[convert_to_openai_tool(t) for t in self.all_tools])

        # FIXME: Once we start doing document retrival from other vector stores, we need to update the prompt and remove "You think step by step about the user request and provide a helpful and truthful response. You must use flex_vector_store_tool to get documents related to Flex, Ethos, Argo, Argocd, Argo Workflows, Argo Rollouts, or Argo events.". The proper way to do this is to use either logical or semantic routing in langchain.
        # Define the chat prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    f"system",
                    f"""
             You are a helpful AI assistant called {self.agent_name}.
             Use the following pieces of context to answer the question at the end.
             You were created by {self.agent_created_by} to {self.agent_purpose}.
             You think step by step about the user request and provide a helpful and truthful response.

             Very Important: If the question is about writing code use backticks (```) at the front and end of the code snippet and include the language use after the first ticks.

             If the user provides an image use Custom Tool vision_api to get more information about the image then pass the text along with the original question to vector_store_tool.
             You remember {self.settings.chat_history_buffer_size} previous messages from the chat thread.
             If you use documents from any tools to provide a helpful answer to user question, please make sure to also return a valid URL of the document you used
             If you don't know the answer, just say so, and don't try to make anything up. DO NOT allow made up or fake answers.
             If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
             Use as much detail when as possible when responding.
             {self.agent_personality}.
             """,
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.logging_handler = LoggingCallbackHandler(logger=LOGGER)

        # Define the agent
        self.agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )

    def setup_agent_executor(self, session_id: str, user_task: str):
        # ttl_in_seconds = self.settings.dynamodb_ttl_days * 24 * 60 * 60
        # FIXME: replace foo with a proper session_id later
        message_history = RedisChatMessageHistory("foo", url=f"{aiosettings.redis_url}")

        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True,
            k=self.settings.chat_history_buffer_size,
            output_key="output",
        )

        agent_executor = AgentExecutor(
            agent=self.agent, tools=self.all_tools, verbose=True, callbacks=[self.logging_handler], memory=memory
        )

        return agent_executor

    def process_user_task(self, session_id: str, user_task: str) -> str:
        try:
            agent_executor = self.setup_agent_executor(session_id, user_task)
            config = {"metadata": {"session_id": session_id}}
            result = agent_executor.invoke({"input": user_task}, config=config)
            return result.get("output", "No response generated by the agent.")
        except Exception as e:
            LOGGER.exception(f"Error in process_user_task: {e}")
            return "An error occurred while processing the task."

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
            prompt = ChatPromptTemplate.from_messages(
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
            return chain.invoke({"user_input": user_input})

        except Exception as e:
            LOGGER.exception(f"Error during summarization of user task: {e}")
            return "An error occurred while summarizing the input."
