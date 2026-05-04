"""Bundle ``ARTIST`` as a package."""

import os

from . import (
    field,
    flux,
    geometry,
    io,
    nurbs,
    optimization,
    raytracing,
    scenario,
    scene,
    util,
)

ARTIST_ROOT = f"{os.sep}".join(__file__.split(os.sep)[:-2])
"""Reference to the root directory of ``ARTIST``."""

__all__ = [
    "field",
    "flux",
    "geometry",
    "io",
    "nurbs",
    "optimization",
    "raytracing",
    "scenario",
    "scene",
    "util",
    "ARTIST_ROOT",
]
