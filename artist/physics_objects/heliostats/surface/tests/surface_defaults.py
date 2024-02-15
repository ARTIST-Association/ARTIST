import os
from yacs.config import CfgNode as CN
from typing import Any

import yacs
from yacs.config import CfgNode as CN

from artist import ARTIST_ROOT

_C = CN()

_C = CN()
_C.TO_OPTIMIZE_NEW = [
    "surface",
]
_C.SHAPE = "real"  # SWITCH FOR HELIOSTAT MODELS: Ideal, Real, Function, Other, NURBS
_C.ROTATION_OFFSET_ANGLE = 0

_C.IDEAL = CN()
_C.IDEAL.POSITION_ON_FIELD = [0, 0, 0]  # in m
_C.IDEAL.NORMAL_VECS = [0, 0, 1]
_C.IDEAL.WIDTH = 4  # in m
_C.IDEAL.HEIGHT = 4  # in m
_C.IDEAL.ROWS = 32
_C.IDEAL.COLS = 32

# Where to aim the heliostat. If `None`, automatically aim at
# `cfg.AC.RECEIVER.CENTER`.
_C.IDEAL.AIM_POINT = None
# Rotational disturbance angles (x, y and z axes) in degrees.
_C.IDEAL.DISTURBANCE_ROT_ANGLES = [0.0, 0.0, 0.0]
_C.IDEAL.FACETS = CN()
_C.IDEAL.FACETS.POSITIONS = [
    [1.0, -1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, 1.0, 0.0],
]
# Relative to `cfg.H.IDEAL.FACETS.POSITIONS`. These also give half of the
# width and height of the heliostat; see STRAL deflectometry data
# format. If a single value, it will be used for all positions.
# Spans in the north direction.
_C.IDEAL.FACETS.SPANS_N = [0.0, 1.0, 0.0]
# Spans in the east direction.
_C.IDEAL.FACETS.SPANS_E = [-1.0, 0.0, 0.0]

# See `cfg.NURBS.FACETS.CANTING` for documentation.
_C.IDEAL.FACETS.CANTING = CN()
_C.IDEAL.FACETS.CANTING.FOCUS_POINT = 0
_C.IDEAL.FACETS.CANTING.ALGORITHM = "standard"


_C.FUNCTION = CN()
_C.FUNCTION.POSITION_ON_FIELD = [0, 0, 0]  # in m
_C.FUNCTION.WIDTH = 4  # in m
_C.FUNCTION.HEIGHT = 4  # in m
_C.FUNCTION.ROWS = 64
_C.FUNCTION.COLS = 64
_C.FUNCTION.NAME = "sin"
_C.FUNCTION.FREQUENCY = 2
_C.FUNCTION.REDUCTION_FACTOR = 1000

# See `cfg.H.IDEAL` for documentation.
_C.FUNCTION.AIM_POINT = None
_C.FUNCTION.DISTURBANCE_ROT_ANGLES = [0.0, 0.0, 0.0]
_C.FUNCTION.FACETS = CN()
_C.FUNCTION.FACETS.POSITIONS = _C.IDEAL.FACETS.POSITIONS.copy()
_C.FUNCTION.FACETS.SPANS_N = _C.IDEAL.FACETS.SPANS_N.copy()
_C.FUNCTION.FACETS.SPANS_E = _C.IDEAL.FACETS.SPANS_E.copy()

_C.FUNCTION.FACETS.CANTING = CN()
_C.FUNCTION.FACETS.CANTING.FOCUS_POINT = 0
_C.FUNCTION.FACETS.CANTING.ALGORITHM = "standard"

_C.DEFLECT_DATA = CN()
# IF `None`, use position from file.
_C.DEFLECT_DATA.POSITION_ON_FIELD = [0, 0, 0]  # in m
_C.DEFLECT_DATA.FILENAME = "Helio_AA39_Rim0_STRAL-Input_211028212814.binp"
_C.DEFLECT_DATA.VERBOSE = True

_C.DEFLECT_DATA.TAKE_N_VECTORS = 8000
_C.DEFLECT_DATA.CONCENTRATORHEADER_STRUCT_FMT = "=5f2I2f"
_C.DEFLECT_DATA.FACETHEADER_STRUCT_FMT = "=i9fI"
_C.DEFLECT_DATA.RAY_STRUCT_FMT = "=7f"

# See `cfg.H.IDEAL` for documentation.
_C.DEFLECT_DATA.AIM_POINT = [0.0, -50.0, 0.0]
_C.DEFLECT_DATA.DISTURBANCE_ROT_ANGLES = [0.0, 0.0, 0.0]
_C.DEFLECT_DATA.FACETS = CN()
# Positions and spans are read from the data.
_C.DEFLECT_DATA.FACETS.CANTING = CN()
_C.DEFLECT_DATA.FACETS.CANTING.FOCUS_POINT = 0.0
_C.DEFLECT_DATA.FACETS.CANTING.ALGORITHM = "standard"

_C.NURBS = CN()
_C.NURBS.MAX_ABS_NOISE = 0.01

_C.NURBS.SPLINE_DEGREE = 3
# Position, width, height, rows, cols (discretization dimensions), and
# facet/canting parameters given by `_C.H.IDEAL`.
# These are again the NURBS rows/cols of the control point matrix.
_C.NURBS.ROWS = 8
_C.NURBS.COLS = 8

_C.OTHER = CN()
_C.OTHER.FILENAME = "tinker.obj"
_C.OTHER.USE_WEIGHTED_AVG = True

# See `cfg.H.IDEAL` for documentation.
_C.OTHER.AIM_POINT = None
_C.OTHER.DISTURBANCE_ROT_ANGLES = [0.0, 0.0, 0.0]
_C.OTHER.FACETS = CN()
_C.OTHER.POSITION_ON_FIELD = [0, 0, 0]  # in m
_C.OTHER.FACETS.POSITIONS = [0.0, 0.0, 0.0]
_C.OTHER.FACETS.SPANS_N = [0.0, float("inf"), 0.0]
_C.OTHER.FACETS.SPANS_E = [-float("inf"), 0.0, 0.0]
_C.OTHER.FACETS.CANTING = CN()
_C.OTHER.FACETS.CANTING.FOCUS_POINT = 0
_C.OTHER.FACETS.CANTING.ALGORITHM = "standard"


def get_cfg_defaults() -> CN:
    return _C.clone()


def load_config_file(cfg: CN) -> CN:
    path = os.path.join(
        ARTIST_ROOT,
        "artist",
        "physics_objects",
        "heliostats",
        "surface",
        "tests",
        "surface_test.yaml",
    )
    cfg.merge_from_file(path)
    return cfg
