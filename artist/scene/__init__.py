"""Bundles all classes that represent the environment in ARTIST."""

from artist.scene.light_source import LightSource
from artist.scene.light_source_array import LightSourceArray
from artist.scene.sun import Sun

__all__ = [
    "LightSourceArray",
    "LightSource",
    "Sun",
]
