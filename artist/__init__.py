"""Bundle ``ARTIST`` as a package."""

import os

from . import core, data_loader, field, scenario, scene, util

ARTIST_ROOT = f"{os.sep}".join(__file__.split(os.sep)[:-2])
"""Reference to the root directory of ARTIST."""

__all__ = [
    "core",
    "data_loader",
    "field",
    "scenario",
    "scene",
    "util",
    "ARTIST_ROOT",
]
