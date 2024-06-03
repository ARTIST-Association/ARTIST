"""Bundles ARTIST as a package."""

import os

from artist import field, raytracing, scenario, scene, util

ARTIST_ROOT = f"{os.sep}".join(__file__.split(os.sep)[:-2])

__all__ = [
    "field",
    "raytracing",
    "scene",
    "util",
    "ARTIST_ROOT",
    "scenario",
]
