"""goob_ai.models.cmds"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from goob_ai import types

# TODO: Current idea is to create a dataclass, one such as:
# DataClassCommand[name: str = "name", command_args = [], command_kargs={}]


@dataclass
class CmdArgs:
    name: str


@dataclass
class DataCmd:
    name: str
    command_args: Union[List[str], None] = []
    command_kargs: Dict[str, str] = {}
