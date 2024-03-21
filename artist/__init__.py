"""This package bundles ARTIST."""

import os

ARTIST_ROOT = f"{os.sep}".join(__file__.split(os.sep)[:-2])

__all__ = [
    "physics_objects",
    "raytracing",
    "scene",
    "util",
    "ARTIST_ROOT",
]
