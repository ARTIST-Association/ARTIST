"""
This package bundles all classes that represent light sources in ARTIST.
"""

from artist.scenario.light_source.light_source import ALightSource
from artist.scenario.light_source.sun import Sun

__all__ = [
    "ALightSource",
    "Sun",
]
