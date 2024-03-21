"""This package bundles ARTIST."""

import os

from artist import physics_objects, raytracing, scene, util

ARTIST_ROOT = f"{os.sep}".join(__file__.split(os.sep)[:-2])

__all__ = ["scene", "physics_objects", "raytracing", "util"]
