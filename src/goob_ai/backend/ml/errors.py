"""goob_ai.backend.ml.errors"""

from __future__ import annotations

import sys
import traceback

from collections.abc import Callable


def run(code: Callable, task: str) -> None:
    """_summary_

    Args:
        code (_type_): _description_
        task (_type_): _description_
    """
    try:
        code()
    except Exception as e:
        print(f"{task}: {type(e).__name__}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
