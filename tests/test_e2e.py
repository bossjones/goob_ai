from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Set, Type

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
from langsmith.evaluation import EvaluationResults, LangChainStringEvaluator, evaluate
from langsmith.run_trees import RunTree
from langsmith.schemas import Example, Run
from loguru import logger as LOGGER

import pytest


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture

import re


# from backend.graph import OPENAI_MODEL_KEY, format_docs, graph

OPENAI_MODEL_KEY = "openai_gpt_4o_mini"

DATASET_NAME = "goob-ai-qa"
EXPERIMENT_PREFIX = "goob-ai-ci"

SCORE_RETRIEVAL_RECALL = "retrieval_recall"
SCORE_ANSWER_CORRECTNESS = "answer_correctness_score"
SCORE_ANSWER_VS_CONTEXT_CORRECTNESS = "answer_vs_context_correctness_score"

# claude sonnet / gpt-4o are a bit too expensive
JUDGE_MODEL_NAME = "claude-3-haiku-20240307"

# judge_llm = ChatAnthropic(model_name=JUDGE_MODEL_NAME)
judge_llm = llm = llm_manager.LlmManager().llm


# Evaluate retrieval

runs = []
examples = []
all_traces = []

all_inputs = []
run_agent_inputs = []


def evaluate_retrieval_recall(run: Run, example: Example) -> dict:
    data = {"evaluation_results": {"runs": run, "examples": example}}
    all_traces.append(data)
    # runs.append(data)
    # examples.append(data)
    # import bpdb

    # bpdb.set_trace()

    documents: list[Document] = run.outputs.get("documents") or []
    sources = [doc.metadata["source"] for doc in documents]
    expected_sources = set(example.outputs.get("sources") or [])
    # NOTE: since we're currently assuming only ~1 correct document per question
    # this score is equivalent to recall @K where K is number of retrieved documents
    score = float(any(source in expected_sources for source in sources))
    return {"key": SCORE_RETRIEVAL_RECALL, "score": score}


# QA Evaluation Schema


class GradeAnswer(BaseModel):
    """Evaluate correctness of the answer and assign a continuous score."""

    reason: str = Field(description="1-2 short sentences with the reason why the score was assigned")
    score: float = Field(
        description="Score that shows how correct the answer is. Use 1.0 if completely correct and 0.0 if completely incorrect",
        minimum=0.0,
        maximum=1.0,
    )


# Evaluate the answer based on the reference answers


QA_SYSTEM_PROMPT = """You are an expert programmer and problem-solver, tasked with grading answers to questions about Opencv, Rich, or Pillow.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements."""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        (
            "human",
            "QUESTION: \n\n {question} \n\n TRUE ANSWER: {true_answer} \n\n STUDENT ANSWER: {answer}",
        ),
    ]
)

qa_chain = QA_PROMPT | judge_llm.with_structured_output(GradeAnswer)
# qa_chain = LangChainStringEvaluator("cot_qa", prepare_data=lambda run, example: {
#         "question": example.inputs["input"],
#         "reference": example.outputs["output"],
#         "prediction": run.outputs["output"],
# })


def evaluate_qa(run: Run, example: Example) -> dict:
    data = {"evaluate_qa": {"runs": run, "examples": example}}
    all_traces.append(data)
    messages = run.outputs.get("messages") or []
    if not messages:
        return {"score": 0.0}

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        return {"score": 0.0}

    score: GradeAnswer = qa_chain.invoke(
        {
            "question": example.inputs["question"],
            "true_answer": example.outputs["answer"],
            "answer": last_message.content,
        }
    )
    return {"key": SCORE_ANSWER_CORRECTNESS, "score": float(score.score)}


# Evaluate the answer based on the provided context

CONTEXT_QA_SYSTEM_PROMPT = """You are an expert programmer and problem-solver, tasked with grading answers to questions about Opencv, Rich, or Pillow.
You are given a question, the context for answering the question, and the student's answer. You are asked to score the student's answer as either CORRECT or INCORRECT, based on the context.

Grade the student answer BOTH based on its factual accuracy AND on whether it is supported by the context. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more information than the true answer, as long as it does not contain any conflicting statements."""

CONTEXT_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXT_QA_SYSTEM_PROMPT),
        (
            "human",
            "QUESTION: \n\n {question} \n\n CONTEXT: {context} \n\n STUDENT ANSWER: {answer}",
        ),
    ]
)

context_qa_chain = CONTEXT_QA_PROMPT | judge_llm.with_structured_output(GradeAnswer)


def evaluate_qa_context(run: RunTree, example: Example) -> dict:
    run_agent_inputs.append(run)
    data = {"evaluate_qa_context": {"runs": run, "examples": example}}
    all_traces.append(data)
    messages = run.outputs.get("messages") or []
    if not messages:
        return {"score": 0.0}

    documents = run.outputs.get("documents") or []
    if not documents:
        return {"score": 0.0}

    context = format_docs(documents)

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        return {"score": 0.0}

    score: GradeAnswer = context_qa_chain.invoke(
        {
            "question": example.inputs["question"],
            "context": context,
            "answer": last_message.content,
        }
    )
    return {"key": SCORE_ANSWER_VS_CONTEXT_CORRECTNESS, "score": float(score.score)}


# Run evaluation


# def run_graph(inputs: dict[str, Any], model_name: str) -> dict[str, Any]:
#     results = graph.invoke(
#         {"messages": [("human", inputs["question"])]},
#         config={"configurable": {"model_name": model_name}},
#     )
#     return results


def build_agent() -> AgentExecutor:
    agent = AiAgent()
    agent_executor = agent.setup_agent_executor("mock-session-id", "")
    return agent_executor


def run_agent(inputs: dict[str, Any], model_name: str) -> dict[str, Any]:
    all_inputs.append(inputs)
    LOGGER.error(f"inputs: {inputs}")
    LOGGER.error(f"model_name: {model_name}")
    # LOGGER.error(f'inputs["input"]["question"]: {inputs["input"]["question"]}')
    # Convert the messages into LangChain format
    # messages = [payload["inputs"] for payload in inputs["input"]]
    # convert the inputs into what we expect
    messages = []
    for key in inputs["inputs"]:
        messages.append({"input": inputs["inputs"][key]})
    LOGGER.error(f"messages: {messages}")
    # BEFORE:
    #     {
    #     'inputs': {
    #         'input': "{'question': 'Using readthedocs, What the minimum version of python needed to install rich?'}"
    #     }
    # }

    # AFTER:
    # {'input': "{'question': 'Using readthedocs, What the minimum version of python needed to install rich?'}"}

    LOGGER.error(f"messages: {messages}")
    LOGGER.error(f"messages: {messages}")
    LOGGER.error(f"messages: {messages}")
    agent_executor = build_agent()
    results = agent_executor.invoke(messages)
    return results


def evaluate_model(*, model_name: str):
    # import bpdb
    # bpdb.set_trace()
    results = evaluate(
        lambda inputs: run_agent(inputs, model_name=model_name),
        data=DATASET_NAME,
        # evaluators=[evaluate_retrieval_recall, evaluate_qa, evaluate_qa_context],
        # DISABLED: one at a time
        evaluators=[evaluate_qa_context],
        experiment_prefix=EXPERIMENT_PREFIX,
        metadata={
            "model_name": model_name,
            "judge_model_name": JUDGE_MODEL_NAME,
            "langchain_version": version("langchain"),
            "langchain_community_version": version("langchain_community"),
            "langchain_core_version": version("langchain_core"),
            "langchain_openai_version": version("langchain_openai"),
            "langchain_text_splitters_version": version("langchain_text_splitters"),
            "langsmith_version": version("langsmith"),
            "pydantic_version": version("pydantic"),
            "pydantic_settings_version": version("pydantic_settings"),
        },
        max_concurrency=4,
    )
    return results


# Check results


def convert_single_example_results(evaluation_results: EvaluationResults):
    converted = {}
    for r in evaluation_results["results"]:
        converted[r.key] = r.score
    return converted


# @pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.integration()
@pytest.mark.evals()
@pytest.mark.slow()
@pytest.mark.flaky()
# NOTE: this is more of a regression test
def test_scores_regression(caplog: LogCaptureFixture, capsys: CaptureFixture):
    # import bpdb
    # bpdb.set_trace()
    # test most commonly used model
    experiment_results = evaluate_model(model_name=OPENAI_MODEL_KEY)
    experiment_result_df = pd.DataFrame(
        convert_single_example_results(result["evaluation_results"]) for result in experiment_results._results
    )
    average_scores = experiment_result_df.mean()

    assert average_scores[SCORE_RETRIEVAL_RECALL] >= 0.65
    assert average_scores[SCORE_ANSWER_CORRECTNESS] >= 0.9
    assert average_scores[SCORE_ANSWER_VS_CONTEXT_CORRECTNESS] >= 0.9

    out, err = capsys.readouterr()

    # Get the exact url of the trace that was just run
    # Get the exact url of the trace that was just run
    with capsys.disabled():
        pattern = r".*View the evaluation results for experiment: '.*?' at:\s+https:\/\/(?:adobe-platform-langsmith-poc-[\w\-]+|smith\.langchain\.com)[\w\-\.\/\?\=\&\%]*"
        match = re.findall(pattern, out)
        print(match)
