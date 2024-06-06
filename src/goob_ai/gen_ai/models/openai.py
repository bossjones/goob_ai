"""goob_ai.gen_ai.models.openai: This module contains the OpenAI language model (LLM) instance for the Goob AI application."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from goob_ai.aio_settings import aiosettings


OPENAI_LLM = ChatOpenAI(
    model=aiosettings.chat_model, temperature=aiosettings.llm_temperature, api_key=aiosettings.openai_api_key
)
