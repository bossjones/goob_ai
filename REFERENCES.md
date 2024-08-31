# Working examples

## Chat Bots

- <https://github.com/langchain-ai/weblangchain/blob/main/main.py>
- <https://python.langchain.com/docs/use_cases/chatbots/>
- <https://python.langchain.com/docs/use_cases/question_answering/>
- <https://github.com/Kav-K/GPTDiscord/tree/0e16004b6bc74812e186137c331f7b9e63882750/cogs>
- <https://github.com/CrosswaveOmega/NikkiBot/blob/7092ae6da21c86c7686549edd5c45335255b73ec/cogs/AiCog.py>

## Vision API GPT 4

- <https://medium.com/@bpothier/generating-structured-data-from-an-image-with-gpt-vision-and-langchain-34aaf3dcb215>

## Example of Discord.bot + fastapi

- <https://github.com/Haste171/langchain-chatbot/blob/main/app.py>

## vectorstores

- <https://github.com/wmawhinney1990/DiscordBot/blob/master/aibot/items/aidoc.py>

## send message if too long discord

- <https://github.com/namastexlabs/namastex_insights/blob/5cb75699ac48c7c2715c20ad58ff4a9c3d3a0ddd/experimental/bot.py#L117-L129>

## Auto use aider to write pytests

example session:

```
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
> /add src/goob_ai/utils/timer.py

Added /Users/malcolm/dev/bossjones/goob_ai/src/goob_ai/utils/timer.py to the chat
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
src/goob_ai/utils/timer.py
> /add tests/utils/test_timer.py

Added /Users/malcolm/dev/bossjones/goob_ai/tests/utils/test_timer.py to the chat
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
src/goob_ai/utils/timer.py tests/utils/test_timer.py
> using pytest, write tests for test_timer.py that cover every function in timer.py. mock if necessary but only use pytest-mock, do not use unittest.mock.

Here are the tests for test_timer.py that cover every function in timer.py using pytest and pytest-mock:

tests/utils/test_timer.py


 <<<<<<< SEARCH
 =======
 import pytest
 import time
 from src.goob_ai.utils.timer import Timer, TimerCollection

 def test_timer_start_stop(mocker):
     timer = Timer()
     mock_time = mocker.patch("time.time", side_effect=[1, 2])
     timer.start()
     timer.stop()
     assert timer.duration() == 1.0

 def test_timer_duration_running(mocker):
     timer = Timer()
     mock_time = mocker.patch("time.time", side_effect=[1, 2, 3])
     timer.start()
     assert timer.duration() == 1.0
     timer.stop()
     assert timer.duration() == 2.0

 def test_timer_reset(mocker):
     timer = Timer()
     mock_time = mocker.patch("time.time", side_effect=[1, 2])
     timer.start()
     timer.stop()
     timer.reset()
     assert timer.duration() == 0.0

 def test_timer_str(mocker):
     timer = Timer()
     mock_time = mocker.patch("time.time", side_effect=[1, 2])
     timer.start()
     timer.stop()
     assert str(timer) == "1.000s"

 def test_timer_collection_start_stop(mocker):
     tc = TimerCollection()
     mock_time = mocker.patch("time.time", side_effect=[1, 2, 3, 4])
     tc.start("Timer 1")
     tc.start("Timer 2")
     tc.stop("Timer 1")
     tc.stop("Timer 2")
     assert tc.duration("Timer 1") == 1.0
     assert tc.duration("Timer 2") == 2.0

 def test_timer_collection_reset(mocker):
     tc = TimerCollection()
     mock_time = mocker.patch("time.time", side_effect=[1, 2, 3])
     tc.start("Timer 1")
     tc.stop("Timer 1")
     tc.reset("Timer 1")
     assert tc.duration("Timer 1") == 0.0

 def test_timer_collection_reset_all(mocker):
     tc = TimerCollection()
     mock_time = mocker.patch("time.time", side_effect=[1, 2, 3, 4])
     tc.start("Timer 1")
     tc.start("Timer 2")
     tc.stop("Timer 1")
     tc.stop("Timer 2")
     tc.reset_all()
     assert tc.duration("Timer 1") == 0.0
     assert tc.duration("Timer 2") == 0.0

 def test_timer_collection_names():
     tc = TimerCollection()
     tc.start("Timer 1")
     tc.start("Timer 2")
     assert set(tc.names()) == {"Timer 1", "Timer 2"}

 def test_timer_collection_str(mocker):
     tc = TimerCollection()
     mock_time = mocker.patch("time.time", side_effect=[1, 2, 3, 4])
     tc.start("Timer 1")
     tc.start("Timer 2")
     tc.stop("Timer 1")
     tc.stop("Timer 2")
     assert str(tc) == "Timer 1: 1.000000s\nTimer 2: 2.000000s"
 >>>>>>> REPLACE



Executing <Task finished name='Task-3941' coro=<Logging.async_success_handler() done, defined at /Users/malcolm/.pyenv/versions/3.10.12/envs/aider3/lib/python3.10/site-packages/litellm/utils.py:2112> result=None created at /Users/malcolm/.pyenv/versions/3.10.12/lib/python3.10/asyncio/tasks.py:636> took 0.359 seconds
Did not apply edit to tests/utils/test_timer.py (--dry-run)
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
src/goob_ai/utils/timer.py tests/utils/test_timer.py
```

example prompt:

`sing pytest, write tests for test_timer.py that cover every function in timer.py. mock if necessary but only use pytest-mock, do not use unittest.mock.`

source: <https://www.youtube.com/watch?v=AeASAsPp9LE>

# july 2024

- <https://medium.com/langchain-0-2-insights/langchain-0-2-insights-building-a-versatile-cli-assistant-with-langchain-and-langgraph-bc81a4570ba2>
- <https://github.com/BellaBe/multi-agents-base-cli>
- <https://console.groq.com/docs/quickstart>
- <https://cookbook.openai.com/examples/tag_caption_images_with_gpt4v>
- https://github.com/commitizen-tools/commitizen-action
- https://github.com/humblFINANCE/humblAPI/blob/master/pyproject.toml
- https://github.com/humblFINANCE/humblAPI/blob/master/pyproject.toml
- https://github.com/humblFINANCE/humblAPI/blob/master/pyproject.toml

# aug 2024

- [simonw/shot-scraper - A command-line utility for taking automated screenshots of websites](https://github.com/simonw/shot-scraper)
- https://github.com/langchain-ai/langchain/blob/master/cookbook/tool_call_messages.ipynb?ref=blog.langchain.dev
- https://github.com/langchain-ai/langchain/tree/master/cookbook
- https://github.com/codingjoe/relint - `Write your own linting rules using regular expressions.`
- https://github.com/ionelmc/python-manhole/ - `Debugging manhole for python applications.`
- https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb
- https://github.com/SAMAD101/Chino/blob/e38f3d9d38702beaed37229f66d79e86a7acab26/src/chino/query.py (write a query
    module maybe)
- https://github.com/Dicklesworthstone/llm_aided_ocr
- https://news.ycombinator.com/item?id=41203306
- https://github.com/dabble-of-devops-bioanalyze/aws-bedrock-utilities/blob/2c3e6effd0d335dcd67d555dcdad88736944fa27/aws_bedrock_utilities/models/pgvector_knowledgebase.py#L21
    - good example of how to use pgvector. insert_pdf_embeddings. load_and_split_pdf. get_loader.

______________________________________________________________________

# Advanced rag suggestions

> https://www.reddit.com/r/LangChain/comments/1cyjfap/best_stack_for_rag/

### Quotes

- If I had to do it over again, I'd just put everything in Postgres with pgvector turned on.
- 100%. It's extremely powerful and it's nice when you have it mixed in with conventional database tables. You can do
    joins across relational and vector data. The performance of the vector indexing database engine will never be a
    significant performance bottleneck. Performance is mostly affected by the embedding model, LLM, and how many tokens
    the agent library (e.g. langchain) uses. Besides, Postgres is no slouch when it comes to performance and is easy to
    scale.
-

### Postgres + pgvector

```sql

# The following statement retrieves the database names from the pg_database view:
SELECT datname FROM pg_database;

# Showing tables using pg_catalog schema - The following statement retrieves the table in PostgreSQL from the pg_catalog.pg_tables view:
SELECT *
FROM pg_catalog.pg_tables
WHERE schemaname != 'pg_catalog' AND
    schemaname != 'information_schema';




# 2) PostgreSQL DESCRIBE TABLE using information_schema
The information_schema.columns catalog contains the information on columns of all tables. To get information on columns of a table, you query the information_schema.columns catalog.
SELECT
  column_name,
  data_type,
  character_maximum_length,
  is_nullable,
  column_default
FROM
  information_schema.columns
WHERE
  table_name = 'langchain_pg_collection';

SELECT
  column_name,
  data_type,
  character_maximum_length,
  is_nullable,
  column_default
FROM
  information_schema.columns
WHERE
  table_name = 'langchain_pg_embedding';
```

______________________________________________________________________

# via cursor:

> Here is a markdown file with examples of practical psql commands, based on the PostgreSQL tutorial you provided:

# Practical psql Commands

## Connecting to a PostgreSQL Database

To connect to a specific database under a user:

```bash
$ psql -d database -U user -W
```

PostgreSQL will prompt for the user's password.

To connect to a remote PostgreSQL instance:

```bash
$ psql -U username -h hostname -p port -d database
```

For example:

```bash
$ psql -U doadmin -h production-sfo-test1-do-user-4866002-0.db.ondigitalocean.com -p 25060 -d defaultdb
```

Enter the password when prompted. \[1\]

## Getting Information About Databases

To list available databases:

```sql
\l
```

To describe a table:

```sql
\d table_name
```

To list available schemas:

```sql
\dn
```

To list available functions:

```sql
\df
```

To execute the previous command again:

```sql
SELECT version();
\g
```

\[1\]

## Timing Queries

To turn on query execution time:

```sql
\timing
SELECT count(*) FROM table;
```

The execution time will be displayed after the query results. \[1\]

## Maintenance

To locate the `pg_hba.conf` configuration file:

```sql
show hba_file;
```

To reload the `pg_hba.conf` configuration file while Postgres is running:

```sql
SELECT pg_reload_conf();
```

\[1\]

## References

1. https://tomcam.github.io/postgres/

The key points are:

- Use `psql` to interactively work with PostgreSQL databases
- `\l` lists databases
- `\d table` describes a table
- `\dn` lists schemas
- `\df` lists functions
- `\timing` turns on query timing
- Configuration is in `pg_hba.conf`, location found via `show hba_file`
- `pg_hba.conf` can be reloaded without restarting using `SELECT pg_reload_conf()`

Let me know if you have any other questions!
