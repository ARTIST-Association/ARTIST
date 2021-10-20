from yacs.config import CfgNode as CN
import os

_C = CN()
# UNIQUE EXPERIMENT IDENTIFIER
_C.ID                                   = 'LongRunWeekend1'
_C.LOGDIR                               = 'Images'
_C.SEED                                 = 0
_C.USE_GPU                              = True
_C.USE_CURL                             = False
_C.USE_NURBS                            = True
_C.CP_PATH                              = ""

# NURBS settings

_C.NURBS                                = CN()
# If setting this to `False`, be aware that the NURBS surface will
# always be evaluated at each surface position independently of the ray
# origins.
_C.NURBS.SET_UP_WITH_KNOWLEDGE          = True
_C.NURBS.FIX_SPLINE_CTRL_WEIGHTS        = True
_C.NURBS.OPTIMIZE_Z_ONLY                = True
_C.NURBS.RECALCULATE_EVAL_POINTS        = True
_C.NURBS.SPLINE_DEGREE                  = 3

# H = Heliostat
_C.H                                    = CN()
_C.H.POSITION_ON_FIELD                  = [0, 0, 0] # in m
# Possible Modes: "Ideal", "Real", "Function", "Other"
# "Real" For Deflec Data
_C.H.SHAPE                              = "Function"

_C.H.IDEAL                              = CN()
_C.H.IDEAL.NORMAL_VECS                  = [0, 0, 1]
_C.H.IDEAL.WIDTH                        = 4 # in m
_C.H.IDEAL.HEIGHT                       = 4 # in m
_C.H.IDEAL.ROWS                         = 32
_C.H.IDEAL.COLS                         = 32

_C.H.FUNCTION                           = CN()
_C.H.FUNCTION.WIDTH                     = 4 # in m
_C.H.FUNCTION.HEIGHT                    = 4 # in m
_C.H.FUNCTION.ROWS                      = 32
_C.H.FUNCTION.COLS                      = 32
_C.H.FUNCTION.NAME                      = "sin"
_C.H.FUNCTION.REDUCTION_FACTOR          = 1000

_C.H.NURBS                              = CN()
_C.H.NURBS.NORMAL_VECS                  = [0, 0, 1]
_C.H.NURBS.WIDTH                        = 4 # in m
_C.H.NURBS.HEIGHT                       = 4 # in m
_C.H.NURBS.ROWS                         = 20
_C.H.NURBS.COLS                         = 20

_C.H.DEFLECT_DATA                       = CN()
_C.H.DEFLECT_DATA.FILENAME              = "Helio_AA33_Rim0_STRAL-Input.binp"
_C.H.DEFLECT_DATA.TAKE_N_VECTORS        = 1000
_C.H.DEFLECT_DATA.CONCENTRATORHEADER_STRUCT_FMT = '=5f2I2f'
_C.H.DEFLECT_DATA.FACETHEADER_STRUCT_FMT        = '=i9fI'
_C.H.DEFLECT_DATA.RAY_STRUCT_FMT                = '=7f'

_C.H.REAL                               = CN()
_C.H.OTHER                              = CN()
_C.H.OTHER.FILENAME                     = ''
_C.H.OTHER.USE_WEIGHTED_AVG             = True

# TODO add heliostat up vec ("rotation")

# AC = Ambiant Conditions
_C.AC                                   = CN()
_C.AC.RECEIVER                          = CN()
# in m in global coordinates
_C.AC.RECEIVER.CENTER                   = [-325, 0, 0]
_C.AC.RECEIVER.PLANE_NORMAL             = [1, 0, 0] # NWU
_C.AC.RECEIVER.PLANE_X                  = 10 # in m
_C.AC.RECEIVER.PLANE_Y                  = 10 # in m
# These X and Y are height and width respectively.
_C.AC.RECEIVER.RESOLUTION_X             = 128
_C.AC.RECEIVER.RESOLUTION_Y             = 128

_C.AC.SUN                               = CN()
_C.AC.SUN.ORIGIN                        = [[-1, 0, 0.1],
                                            [-1,-1,0.1],
                                            [1,0,0.1],
                                            [1,1,0.1],
                                            [0,1,0.1],
                                            [0,0.1,1],
                                            [1,1,1]
                                            ]
_C.AC.SUN.GENERATE_N_RAYS               = 1000
_C.AC.SUN.DISTRIBUTION                  = "Normal"

_C.AC.SUN.NORMAL_DIST                   = CN()
_C.AC.SUN.NORMAL_DIST.MEAN              = [0,0]
_C.AC.SUN.NORMAL_DIST.COV               = [[0.002090**2, 0], [0, 0.002090**2]]

_C.TRAIN_PARAMS                         = CN()
_C.TRAIN_PARAMS.EPOCHS                  = 60000


def get_cfg_defaults():
    return _C.clone()


def load_config_file(cfg, config_file_loc):
    if len(os.path.splitext(config_file_loc)[1]) == 0:
        config_file_loc += '.yaml'
    cfg.merge_from_file(config_file_loc)
    
    # if experiment_name:
    #     cfg.merge_from_list(["ID", experiment_name])

    return cfg
