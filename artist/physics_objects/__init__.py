"""This package bundles all classes that represent physical objects in ARTIST."""

from artist.physics_objects import heliostats
from artist.physics_objects.module import AModule

__all__ = [
    "AModule",
    "heliostats",
]
