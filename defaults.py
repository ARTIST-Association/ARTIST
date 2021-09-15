from yacs.config import CfgNode as CN
import os

_C = CN()
_C.ID                                   = 'Default' # UNIQUE EXPERIMENT IDENTIFIER
_C.LOGDIR                               = 'Images'
_C.SEED                                 = 0
_C.USE_GPU                              = True
_C.USE_CURL                             = False
_C.USE_NURBS                            = False # If setting this to `False`, be aware that the NURBS surface will
                                                # always be evaluated at each surface position independently of the ray
                                                # origins.



# NURBS settings

_C.NURBS                                = CN()
_C.NURBS.SET_UP_WITH_KNOWLEDGE          = True
_C.NURBS.FIX_SPLINE_CTRL_WEIGHTS        = True
_C.NURBS.SPLINE_DEGREE                  = 2

_C.H                                    = CN() # H = Heliostat
_C.H.POSITION_ON_FIELD                  = [0,0,0] # in m
_C.H.SHAPE                              = "Ideal" # Possible Modes: "Ideal", "Real", "Other" || "Real" For Deflec Data

_C.H.IDEAL                              = CN()
_C.H.IDEAL.NORMAL_VECS                  = [0,0,1]
_C.H.IDEAL.WIDTH                        = 4 # in m
_C.H.IDEAL.HEIGHT                       = 4 # in m
_C.H.IDEAL.ROWS                         = 32
_C.H.IDEAL.COLS                         = 32

_C.H.NURB                              = CN()
_C.H.NURB.NORMAL_VECS                  = [0,0,1]
_C.H.NURB.WIDTH                        = 4 # in m
_C.H.NURB.HEIGHT                       = 4 # in m
_C.H.NURB.ROWS                         = 32
_C.H.NURB.COLS                         = 32

_C.H.DEFLECT_DATA                       = CN()
_C.H.DEFLECT_DATA.FILENAME              = "NoFilename"
_C.H.DEFLECT_DATA.TAKE_N_VECTORS        = 1000
_C.H.CONCENTRATORHEADER_STRUCT_FMT      = '=5f2I2f'
_C.H.FACETHEADER_STRUCT_FMT             = '=i9fI'
_C.H.RAY_STRUCT_FMT                     = '=7f'

_C.H.REAL                               = CN()
_C.H.OTHER                              = CN()

# TODO add heliostat up vec ("rotation")

_C.AC                                   = CN() # AC = Ambiant Conditions
_C.AC.RECEIVER                          = CN()
_C.AC.RECEIVER.CENTER                   = [-25,0,0] # in m in global coordinates
_C.AC.RECEIVER.PLANE_NORMAL             = [1,0,0] # NWU
_C.AC.RECEIVER.PLANE_X                  = 10 # in m
_C.AC.RECEIVER.PLANE_Y                  = 10 # in m
_C.AC.RECEIVER.RESOLUTION_X             = 256
_C.AC.RECEIVER.RESOLUTION_Y             = 256

_C.AC.SUN                               = CN()
_C.AC.SUN.ORIGIN                        = [-1,0,0]
_C.AC.SUN.GENERATE_N_RAYS               = 1000
_C.AC.SUN.DISTRIBUTION                  = "Normal"

_C.AC.SUN.NORMAL_DIST                   = CN()
_C.AC.SUN.NORMAL_DIST.MEAN              = [0,0]
_C.AC.SUN.NORMAL_DIST.COV               = [[0.002090**2, 0], [0, 0.002090**2]]



_C.TRAIN_PARAMS                         = CN()
_C.TRAIN_PARAMS.EPOCHS                  = 2000







def get_cfg_defaults():
    return _C.clone()


def load_config_file(cfg, config_file_loc, experiment_name=None):
    if len(os.path.splitext(config_file_loc)[1]) == 0:
        config_file_loc += '.yaml'
    cfg.merge_from_file(config_file_loc)
    if experiment_name:
        cfg.merge_from_list(["ID", experiment_name])

    return cfg
