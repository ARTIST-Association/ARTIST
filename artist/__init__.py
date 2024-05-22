"""Bundles ARTIST as a package."""

import os

from .scenario import Scenario

ARTIST_ROOT = f"{os.sep}".join(__file__.split(os.sep)[:-2])

__all__ = [
    "field",
    "raytracing",
    "scene",
    "util",
    "ARTIST_ROOT",
    "Scenario",
]
