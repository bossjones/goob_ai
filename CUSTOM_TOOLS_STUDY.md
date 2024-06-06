# Examples of custom tools

## StudyTime

https://github.com/f-andrei/StudyTime/blob/b8e1319569ceaeaf0239944eaecc9dd2392b4880/app/chatbot/custom_tools.py#L7

```python
from langchain.memory import ConversationBufferMemory
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import Tool
from langchain_core.tools import BaseTool
from typing import Type,  Any
from utils.embed_utils import display_embed
import sqlite3
import discord
import asyncio
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = 'database'
DB_NAME = 'studytime.sqlite3'
DB_FILE = os.path.join(ROOT_DIR, DB_DIR, DB_NAME)


memory = ConversationBufferMemory(memory_key="chat_history")

class QueryDataInput(BaseModel):
    query: str = Field(..., description="SQLite3 query (sql)")

    class Config:
        arbitrary_types_allowed = True

class QueryData(BaseTool):
    name: str = "query_data"
    args_schema: Type[BaseModel] = QueryDataInput
    description: str =  f"""A SQLite query executor. Useful for when you need to answer
    about user's tasks and notes. Input should be a SQLite3 command."""

    def _run(self, query) -> tuple:
        try:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()

                # Execute the query
                cursor.execute(query)
                results = cursor.fetchall()

                # Get the last row id
                last_row_id = cursor.lastrowid

                return f"Results: {results}, Last row id:{last_row_id}"
        except sqlite3.Error as e:
            return None, f"Query failed with error: {e}"


class DatabaseInfo(BaseTool):
    name: str = "database_info"
    description: str = """Retrieves information about a database. Useful
    when you need to answer about user's tasks or notes. No input needed.
    Contains all the relevant structure about a data base (tables, columns,
    additional information)."""

    def _run(self, _=None) -> str:
        try:
            memory.chat_memory.add_ai_message()
        except Exception as e:
            return f"Error occurred while retrieving database information: {e}"

class CreateEmbedInput(BaseModel):
    query_output: Any = Field(..., description="Output from QueryData")


class CreateEmbed(BaseTool):
    name: str = "create_embed"
    args_schema: Type[BaseModel] = CreateEmbedInput
    description: str = """Creates a Discord embed. Useful to format tasks
    or notes in a nicer way. Takes the ouput from QueryData.
    Example: 29, 'Programar', 'Discord Bot', 'https://discord.com', '2024-02-17 02:30', 5.0, 1, 227128911576694784
    id, name, description, links, start_date, duration, user_id
    Send only the values, do not send the keys.
    """
    def _run(self, query_output: str) -> discord.Embed:
        try:
            query_output = query_output.strip('()')
            query_output = query_output.replace("'", '')
            query_output = query_output.split(', ')
            if len(query_output) >= 7:
                embed_type = 'task'
            else:
                embed_type = 'note'
            asyncio.create_task(self.async_display_embed(query_output, embed_type))
            memory.chat_memory.add_ai_message('Embed created')
            return 'Embed successfully created'
        except Exception as e:
            return f"An error occurred while creating an embed: {e}"

    async def async_display_embed(self, query_output, embed_type):
        try:
            # Call the asynchronous function
            embed_type = embed_type.lower()
            result = await display_embed(query_output, type=embed_type)
            return result
        except Exception as e:
            print(f"An error occurred while creating an embed: {e}")

db_info = DatabaseInfo()
db_query = QueryData()
create_embed = CreateEmbed()

tools = [
    Tool(
        name="DatabaseInfo",
        func=db_info.run,
        description="Useful to understand the structure of an sqlite database, such as table names and columns. Takes no arguments."
    ),
    Tool(
        name="QueryData",
        func=db_query.run,
        description="Useful to query data from a structural database. Takes only the SQL query. Adjust the user input so that it matches the database dtypes."
    ),
    Tool(
        name="CreateEmbed",
        func=create_embed.run,
        description="Useful to create embeds for tasks or notes. Takes the output of QueryData."
    )
]

def handle_async_display_embed(query_output, embed_type):
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(display_embed(query_output, embed_type))
    return result
```
