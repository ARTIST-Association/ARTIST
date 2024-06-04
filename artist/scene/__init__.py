"""Bundles all classes that represent the environment in ARTIST."""

print("SCENE IMPORT")
from .light_source import LightSource
from .light_source_array import LightSourceArray
from .sun import Sun

print("SCENE IMPORT FINISHED")

__all__ = [
    "LightSourceArray",
    "LightSource",
    "Sun",
]
