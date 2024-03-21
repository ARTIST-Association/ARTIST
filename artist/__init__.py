"""This package bundles ARTIST."""

import os

from . import physics_objects, raytracing, scene, util

ARTIST_ROOT = f"{os.sep}".join(__file__.split(os.sep)[:-2])

__all__ = [
    "physics_objects",
    "raytracing",
    "scene",
    "util",
    "ARTIST_ROOT",
]
