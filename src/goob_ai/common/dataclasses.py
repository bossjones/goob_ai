from __future__ import annotations

from enum import Enum

from pydantic.dataclasses import dataclass


# Enums for surface values
class SurfaceType(str, Enum):
    DISCORD = "discord"
    UNKNOWN = "unknown"


# Dataclass for surface information using Pydantic's dataclass
@dataclass(config={"use_enum_values": True})
class SurfaceInfo:
    surface: SurfaceType
    type: str
    source: str
