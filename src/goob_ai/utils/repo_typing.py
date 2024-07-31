# SOURCE: https://github.com/drawjk705/us-congress/blob/dedae9171953f770d9b6efe4f1e4db83f60dd4b4/run_pyright.py#L9
from __future__ import annotations

import os
import re

from typing import Set


PYRIGHT_CMD = "pyright -p pyproject.toml ."
MISSING_TYPESTUB_PATTERN = r'.*error: Stub file not found for "(.*)".*'


def run_pyright() -> None:
    """
    Find all missing typestubs, generate them,
    then run pyright
    """
    modulesMissingStubs: set[str] = set()

    for line in os.popen(PYRIGHT_CMD).readlines():
        match = re.match(MISSING_TYPESTUB_PATTERN, line)
        print(f"math: {match}")
        if match:
            group = match[1]
            group = re.sub(r"\..*", "", group)
            modulesMissingStubs.add(group)

    for module in modulesMissingStubs:
        cmd = f"{PYRIGHT_CMD} --createstub {module}"
        print(cmd)
        # os.system(cmd)

    print(PYRIGHT_CMD)
    os.system(PYRIGHT_CMD)


if __name__ == "__main__":
    run_pyright()
