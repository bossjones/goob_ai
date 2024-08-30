from __future__ import annotations

import json

from importlib.metadata import version
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Set, Type

import langsmith
import pandas as pd

from goob_ai import llm_manager
from goob_ai.agent import AiAgent
from goob_ai.tools.rag_tool import format_docs
from langchain.agents import AgentExecutor
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langsmith import Client
from langsmith.evaluation import EvaluationResults, LangChainStringEvaluator, evaluate
from langsmith.run_trees import RunTree
from langsmith.schemas import Example, Run
from loguru import logger as LOGGER


# Load the example inputs from q_a.json
with open("scripts/q_a.json") as file:
    example_inputs = [
        (item["question"], item["answer"])
        for item in json.load(file)
    ]

client = langsmith.Client()
dataset_name = "Climate Change Q&A"

# Storing inputs in a dataset lets us
# run chains and LLMs over a shared set of examples.
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Questions and answers about climate change.",
)
for input_prompt, output_answer in example_inputs:
    client.create_example(
        inputs={"question": input_prompt},
        outputs={"answer": output_answer},
        metadata={"source": "Various"},
        dataset_id=dataset.id,
    )
