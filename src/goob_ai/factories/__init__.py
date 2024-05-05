"""goob_ai.factories"""

from __future__ import annotations

import dataclasses

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


read_only = "read_only"
coerce_to = "coerce_to"


@dataclass
class SerializerFactory:
    def as_dict(self) -> Dict:
        d = dataclasses.asdict(self)
        for f in dataclasses.fields(self):
            if read_only in f.metadata.keys():
                del d[f.name]
            if coerce_to in f.metadata.keys():
                d[f.name] = f.metadata[coerce_to](d[f.name])
        return d
