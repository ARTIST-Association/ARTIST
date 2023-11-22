
from yacs.config import CfgNode as CN

_C = CN()

_C                                = CN()
_C.SET_UP_WITH_KNOWLEDGE          = False
_C.INITIALIZE_WITH_KNOWLEDGE      = False
_C.INITIALIZE_Z_ONLY              = False
_C.FIX_SPLINE_CTRL_WEIGHTS        = True
_C.FIX_SPLINE_KNOTS               = True
_C.OPTIMIZE_Z_ONLY                = True
_C.RECALCULATE_EVAL_POINTS        = False
_C.SPLINE_DEGREE                  = 3
_C.POSITION_ON_FIELD              = 'inherit'  # in m
_C.AIM_POINT                      = 'inherit'
_C.DISTURBANCE_ROT_ANGLES         = [0, 0, 0]
_C.FACETS                         = CN()
_C.FACETS.CANTING                 = CN()
_C.FACETS.CANTING.FOCUS_POINT     = 0.0
_C.FACETS.CANTING.ALGORITHM       = 'inherit'
_C.GROWING                        = CN()
_C.GROWING.INTERVAL               = 0
_C.GROWING.START_ROWS             = 0
_C.GROWING.START_COLS             = 0
_C.GROWING.STEP_SIZE_ROWS         = 0
_C.GROWING.STEP_SIZE_COLS         = 0
_C.WIDTH                          = 4  # in m
_C.HEIGHT                         = 4  # in m
_C.ROWS                           = 7
_C.COLS                           = 7


def get_cfg_defaults() -> CN:
    return _C.clone()


def load_config_file(cfg: CN) -> CN:
    cfg.merge_from_file("artist\physics_objects\heliostats\surface\\tests\\nurbs_test.yaml")
    return cfg
