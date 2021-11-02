from yacs.config import CfgNode as CN
import os

_C = CN()
# UNIQUE EXPERIMENT IDENTIFIER
_C.ID                                   = 'MediumTargetBoosting12Nurbs'
_C.EXPERIMENT_NAME                      = "Receiver64x64"
_C.LOGDIR                               = 'Results'
_C.SEED                                 = 0
_C.USE_GPU                              = True
_C.USE_CURL                             = False
_C.USE_NURBS                            = True
_C.SAVE_RESULTS                         = True
_C.CP_PATH                              = ""
_C.CP_PATH                              = "C:\\Python\\DiffSTRAL\\diff-stral\\Results\\MediumTargetBoosting12Nurbs\\NoBoostingReceiver32x32\\Logfiles\\NURBSHeliostat.pt"
_C.LOAD_OPTIMIZER_STATE                 = False

# NURBS settings
_C.NURBS                                = CN()
# If setting this to `False`, be aware that the NURBS surface will
# always be evaluated at each surface position independently of the ray
# origins.
_C.NURBS.SET_UP_WITH_KNOWLEDGE          = True
# Whether to initialize the control points according to known
# discretized values.
_C.NURBS.INITIALIZE_WITH_KNOWLEDGE      = True
_C.NURBS.FIX_SPLINE_CTRL_WEIGHTS        = True
_C.NURBS.FIX_SPLINE_KNOTS               = True
_C.NURBS.OPTIMIZE_Z_ONLY                = True
_C.NURBS.RECALCULATE_EVAL_POINTS        = False
# 0 turns progressive growing off
_C.NURBS.PROGRESSIVE_GROWING_INTERVAL   = 2
_C.NURBS.SPLINE_DEGREE                  = 3

# H = Heliostat
_C.H                                    = CN()
_C.H.POSITION_ON_FIELD                  = [0, 0, 0] # in m
# Possible Modes: "Ideal", "Real", "Function", "Other"
# "Real" For Deflec Data
_C.H.SHAPE                              = "Function"                            #SWITCH FOR HELIOSTAT MODELS: 

_C.H.IDEAL                              = CN()
_C.H.IDEAL.NORMAL_VECS                  = [0, 0, 1]
_C.H.IDEAL.WIDTH                        = 4 # in m
_C.H.IDEAL.HEIGHT                       = 4 # in m
_C.H.IDEAL.ROWS                         = 32
_C.H.IDEAL.COLS                         = 32

_C.H.FUNCTION                           = CN()
_C.H.FUNCTION.WIDTH                     = 4 # in m
_C.H.FUNCTION.HEIGHT                    = 4 # in m
_C.H.FUNCTION.ROWS                      = 64
_C.H.FUNCTION.COLS                      = 64
_C.H.FUNCTION.NAME                      = "sin"
_C.H.FUNCTION.FREQUENCY                 = 2
_C.H.FUNCTION.REDUCTION_FACTOR          = 500

_C.H.NURBS                              = CN()
_C.H.NURBS.WIDTH                        = 4 # in m
_C.H.NURBS.HEIGHT                       = 4 # in m
_C.H.NURBS.ROWS                         =12
_C.H.NURBS.COLS                         =12

_C.H.DEFLECT_DATA                       = CN()
_C.H.DEFLECT_DATA.FILENAME              = "Helio_AA33_Rim0_STRAL-Input.binp"
_C.H.DEFLECT_DATA.TAKE_N_VECTORS        = 2025
_C.H.DEFLECT_DATA.CONCENTRATORHEADER_STRUCT_FMT = '=5f2I2f'
_C.H.DEFLECT_DATA.FACETHEADER_STRUCT_FMT        = '=i9fI'
_C.H.DEFLECT_DATA.RAY_STRUCT_FMT                = '=7f'

_C.H.REAL                               = CN()
_C.H.OTHER                              = CN()
_C.H.OTHER.FILENAME                     = 'tinker.obj'
_C.H.OTHER.USE_WEIGHTED_AVG             = True

# TODO add heliostat up vec ("rotation")

# AC = Ambiant Conditions
_C.AC                                   = CN()
_C.AC.RECEIVER                          = CN()
# in m in global coordinates
_C.AC.RECEIVER.CENTER                   = [-25, 0, 0]
_C.AC.RECEIVER.PLANE_NORMAL             = [1, 0, 0] # NWU
_C.AC.RECEIVER.PLANE_X                  = 10 # in m
_C.AC.RECEIVER.PLANE_Y                  = 10 # in m
# These X and Y are height and width respectively.
_C.AC.RECEIVER.RESOLUTION_X             = 64
_C.AC.RECEIVER.RESOLUTION_Y             = 64

_C.AC.SUN                               = CN()
_C.AC.SUN.ORIGIN                        = [
                                            # [-1, 0, 0],
                                           [-0.43719268,  0.7004466,   0.564125  ],
                                            ]
_C.AC.SUN.GENERATE_N_RAYS               = 100
_C.AC.SUN.DISTRIBUTION                  = "Normal"                              #SWITCH FOR SOLAR DISTRIBUSTION: Normal, Point, Pillbox (not completly implemented)

_C.AC.SUN.NORMAL_DIST                   = CN()
_C.AC.SUN.NORMAL_DIST.MEAN              = [0,0]
_C.AC.SUN.NORMAL_DIST.COV               = [[0.002090**2, 0], [0, 0.002090**2]]

_C.TRAIN                                = CN()
_C.TRAIN.EPOCHS                         = 60000

_C.TRAIN.SCHEDULER                      = CN()
_C.TRAIN.SCHEDULER.NAME                 = "ReduceOnPlateu"                      #SWITCH FOR SCHEDULER: ReduceOnPLateu, Cyclic, OneCycle

_C.TRAIN.SCHEDULER.ROP                  = CN()
_C.TRAIN.SCHEDULER.ROP.FACTOR           = 0.5
_C.TRAIN.SCHEDULER.ROP.MIN_LR           = 1e-8
_C.TRAIN.SCHEDULER.ROP.PATIENCE         = 100
_C.TRAIN.SCHEDULER.ROP.COOLDOWN         = 200
_C.TRAIN.SCHEDULER.ROP.VERBOSE          = True

_C.TRAIN.SCHEDULER.CYCLIC               = CN()
_C.TRAIN.SCHEDULER.CYCLIC.BASE_LR       = 1e-8
_C.TRAIN.SCHEDULER.CYCLIC.MAX_LR        = 8.3e-5
_C.TRAIN.SCHEDULER.CYCLIC.STEP_SIZE_UP  = 100
_C.TRAIN.SCHEDULER.CYCLIC.CYCLE_MOMENTUM= False
_C.TRAIN.SCHEDULER.CYCLIC.MODE          ="triangular2"

_C.TRAIN.SCHEDULER.ONE_CYCLE            = CN()
_C.TRAIN.SCHEDULER.ONE_CYCLE.MAX_LR     = 6e-5
_C.TRAIN.SCHEDULER.ONE_CYCLE.START_LR   = 1e-10
_C.TRAIN.SCHEDULER.ONE_CYCLE.FINAL_LR   = 1e-8
_C.TRAIN.SCHEDULER.ONE_CYCLE.THREE_PHASE= True


_C.TRAIN.OPTIMIZER                      = CN()
_C.TRAIN.OPTIMIZER.NAME                 = "Adamax"                              #SWITCH FOR OPTIMIZER: Adam, Adamax, AdamW
_C.TRAIN.OPTIMIZER.LR                   = 2e-4
_C.TRAIN.OPTIMIZER.BETAS                = [0.9, 0.999]
_C.TRAIN.OPTIMIZER.EPS                  = 1e-8
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY         = 0

_C.TRAIN.LOSS                           = CN()
_C.TRAIN.LOSS.NAME                      = "L1"                                  #SWITCH FOR LOSS: L1, MSE
_C.TRAIN.LOSS.USE_L1_WEIGHT_DECAY       = True
_C.TRAIN.LOSS.WEIGHT_DECAY_FACTOR       = 0.2




def get_cfg_defaults():
    return _C.clone()


def load_config_file(cfg, config_file_loc):
    if len(os.path.splitext(config_file_loc)[1]) == 0:
        config_file_loc += '.yaml'
    cfg.merge_from_file(config_file_loc)
    
    # if experiment_name:
    #     cfg.merge_from_list(["ID", experiment_name])

    return cfg
