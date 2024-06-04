"""Bundles ARTIST as a package."""

import os

print("TOP LEVEL IMPORT")
from . import field, raytracing, scenario, scene, util

print("TOP LEVEL IMPORT FINISHED")

ARTIST_ROOT = f"{os.sep}".join(__file__.split(os.sep)[:-2])

__all__ = [
    "field",
    "raytracing",
    "scene",
    "util",
    "ARTIST_ROOT",
    "scenario",
]
