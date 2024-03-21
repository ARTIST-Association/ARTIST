"""This package bundles ARTIST."""

import os

from artist import environment, physics_objects, raytracing, util

__all__ = ["environment", "physics_objects", "raytracing", "util"]


ARTIST_ROOT = f"{os.sep}".join(__file__.split(os.sep)[:-2])
