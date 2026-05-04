"""Bundle ``ARTIST`` as a package."""

import os

from . import core, field, io, scenario, scene, util

ARTIST_ROOT = f"{os.sep}".join(__file__.split(os.sep)[:-2])
"""Reference to the root directory of ``ARTIST``."""

__all__ = [
    "core",
    "io",
    "field",
    "scenario",
    "scene",
    "util",
    "ARTIST_ROOT",
]
