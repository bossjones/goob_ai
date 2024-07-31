"""goob_ai.models.loggers"""

# pyright: reportAssignmentType=false
# pyright: strictParameterNoneValue=false
# pylint: disable=no-name-in-module

# SOURCE: https://blog.bartab.fr/fastapi-logging-on-the-fly/
from __future__ import annotations

from typing import Any, ForwardRef, List, Optional

# from pydantic import BaseModel
from langchain.pydantic_v1 import BaseModel


LoggerModel = ForwardRef("LoggerModel")


class LoggerPatch(BaseModel):
    name: str
    level: str


class LoggerModel(BaseModel):
    name: str
    level: Optional[int]
    # children: Optional[List["LoggerModel"]] = None
    # fixes: https://github.com/samuelcolvin/pydantic/issues/545
    children: Optional[list[Any]] = None
    # children: ListLoggerModel = None


LoggerModel.update_forward_refs()
