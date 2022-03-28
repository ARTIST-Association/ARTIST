from yacs.config import CfgNode as CN
import os

_C = CN()
# UNIQUE EXPERIMENT IDENTIFIER

_C.ID                                   = 'NotGiven'
_C.EXPERIMENT_NAME                      = "NotGiven"
_C.LOGDIR                               = 'Results'
_C.SEED                                 = 42
_C.USE_FLOAT64                          = False
_C.USE_GPU                              = True
_C.USE_CURL                             = False
_C.USE_NURBS                            = True
_C.SAVE_RESULTS                         = True
_C.CP_PATH                              = ""
# _C.CP_PATH                              = "C:\\Python\\DiffSTRAL\\diff-stral\\Results\\MoreNURBS\\LongRunMultiSun\\Logfiles\\MultiNURBSHeliostat.pt"
_C.LOAD_OPTIMIZER_STATE                 = False

# NURBS settings
_C.NURBS                                = CN()
# Whether to use the available width and height information to set up
# the NURBS surface.
# If setting this to `False`, be aware that the NURBS surface will
# always be evaluated at each surface position independently of the ray
# origins.
_C.NURBS.SET_UP_WITH_KNOWLEDGE          = False
# Whether to initialize the control points according to known,
# ideal discretized values.
_C.NURBS.INITIALIZE_WITH_KNOWLEDGE      = False
# Only relevant when `INITIALIZE_WITH_KNOWLEDGE`.
# Whether to only change z values in that initialization step.
_C.NURBS.INITIALIZE_Z_ONLY              = False
_C.NURBS.FIX_SPLINE_CTRL_WEIGHTS        = True
_C.NURBS.FIX_SPLINE_KNOTS               = True
_C.NURBS.OPTIMIZE_Z_ONLY                = True
_C.NURBS.RECALCULATE_EVAL_POINTS        = False
_C.NURBS.SPLINE_DEGREE                  = 3

# For multi-NURBS heliostat
_C.NURBS.FACETS = CN()
_C.NURBS.FACETS.CANTING = CN()
_C.NURBS.FACETS.CANTING.ENABLED = True
# Canting algorithm can be 'standard' or 'active'.
# - Standard canting calculates the canting rotation to the focus point
#   once at the beginning. The focus point is assumed to be right above
#   the heliostat center at the distance of the receiver.
# - In active canting, each facet is canted perfectly for each sun
#   position.
_C.NURBS.FACETS.CANTING.ALGORITHM = 'standard'

# NURBS progressive growing
_C.NURBS.GROWING                        = CN()
# 0 turns progressive growing off
_C.NURBS.GROWING.INTERVAL               = 0
# 0 starts with minimum size
_C.NURBS.GROWING.START_ROWS             = 0
_C.NURBS.GROWING.START_COLS             = 0
# 0 grows a new index between all old ones
_C.NURBS.GROWING.STEP_SIZE_ROWS         = 0
_C.NURBS.GROWING.STEP_SIZE_COLS         = 0

_C.NURBS.WIDTH                        = 4 # in m
_C.NURBS.HEIGHT                       = 4 # in m
# Both of these are used per facet!
_C.NURBS.ROWS                         = 6
_C.NURBS.COLS                         = 6

# H = Heliostat
_C.H                                    = CN()
_C.H.POSITION_ON_FIELD                  = [0, 0, 0] # in m




_C.H.SHAPE                              = "function"                            #SWITCH FOR HELIOSTAT MODELS: Ideal, Real, Function, Other, NURBS

_C.H.IDEAL                              = CN()
_C.H.IDEAL.NORMAL_VECS                  = [0, 0, 1]
_C.H.IDEAL.WIDTH                        = 4 # in m
_C.H.IDEAL.HEIGHT                       = 4 # in m
_C.H.IDEAL.ROWS                         = 32
_C.H.IDEAL.COLS                         = 32

_C.H.IDEAL.FACETS = CN()
_C.H.IDEAL.FACETS.POSITIONS = [
    [-1.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, -1.0, 0.0],
]
# Relative to `cfg.H.IDEAL.FACETS.POSITIONS`. These also give half of the
# width and height of the heliostat; see STRAL deflectometry data
# format. If a single value, it will be used for all positions.
# Spans in the north direction.
_C.H.IDEAL.FACETS.SPANS_X = [0.0, 1.0, 0.0]
# Spans in the east direction.
_C.H.IDEAL.FACETS.SPANS_Y = [-1.0, 0.0, 0.0]

# See `cfg.NURBS.FACETS.CANTING` for documentation.
_C.H.IDEAL.FACETS.CANTING = CN()
_C.H.IDEAL.FACETS.CANTING.ENABLED = False
_C.H.IDEAL.FACETS.CANTING.ALGORITHM = 'standard'


_C.H.FUNCTION                           = CN()
_C.H.FUNCTION.WIDTH                     = 4 # in m
_C.H.FUNCTION.HEIGHT                    = 4 # in m
_C.H.FUNCTION.ROWS                      = 64
_C.H.FUNCTION.COLS                      = 64
_C.H.FUNCTION.NAME                      = "sin"
_C.H.FUNCTION.FREQUENCY                 = 2
_C.H.FUNCTION.REDUCTION_FACTOR          = 1000

_C.H.FUNCTION.FACETS = CN()
# See `cfg.H.IDEAL.FACETS` for documentation.
_C.H.FUNCTION.FACETS.POSITIONS = _C.H.IDEAL.FACETS.POSITIONS.copy()
_C.H.FUNCTION.FACETS.SPANS_X = _C.H.IDEAL.FACETS.SPANS_X.copy()
_C.H.FUNCTION.FACETS.SPANS_Y = _C.H.IDEAL.FACETS.SPANS_Y.copy()
# _C.H.FUNCTION.FACETS.POSITIONS = [
#     [-0.6424999833106995, -0.8075000047683716, 0.040198374539613724],
#     [-0.6424999833106995, 0.8075000047683716, 0.040198374539613724],
#     [0.6424999833106995, -0.8075000047683716, 0.040198374539613724],
#     [0.6424999833106995, 0.8075000047683716, 0.040198374539613724],
# ]
# _C.H.FUNCTION.FACETS.SPANS_X = [
#     [0.0, 0.8024845123291016, -0.004984567407518625],
#     [0.0, 0.8024845123291016, 0.004984567407518625],
#     [0.0, 0.8024845123291016, -0.004984567407518625],
#     [0.0, 0.8024845123291016, 0.004984567407518625],
# ]
# _C.H.FUNCTION.FACETS.SPANS_Y = [
#     [-0.6374922394752502, 1.9569215510273352e-05, 0.0031505227088928223],
#     [-0.6374922394752502, -1.9569215510273352e-05, 0.0031505227088928223],
#     [-0.6374922394752502, -1.9569215510273352e-05, -0.0031505227088928223],
#     [-0.6374922394752502, 1.9569215510273352e-05, -0.0031505227088928223],
# ]

_C.H.FUNCTION.FACETS.CANTING = CN()
_C.H.FUNCTION.FACETS.CANTING.ENABLED = False
_C.H.FUNCTION.FACETS.CANTING.ALGORITHM = 'standard'

_C.H.DEFLECT_DATA                       = CN()
_C.H.DEFLECT_DATA.FILENAME              = "Helio_AA39_Rim0_STRAL-Input_211028212814.binp"
_C.H.DEFLECT_DATA.ZS_PATH               = "Helio_AA39_Rim0_LocalResults_220303111914.csv"
_C.H.DEFLECT_DATA.VERBOSE               = True

_C.H.DEFLECT_DATA.TAKE_N_VECTORS        = 8000
_C.H.DEFLECT_DATA.CONCENTRATORHEADER_STRUCT_FMT = '=5f2I2f'
_C.H.DEFLECT_DATA.FACETHEADER_STRUCT_FMT        = '=i9fI'
_C.H.DEFLECT_DATA.RAY_STRUCT_FMT                = '=7f'

# See `cfg.H.IDEAL.FACETS` for documentation.
_C.H.DEFLECT_DATA.FACETS = CN()
# Positions and spans are read from the data.
_C.H.DEFLECT_DATA.FACETS.CANTING = CN()
_C.H.DEFLECT_DATA.FACETS.CANTING.ENABLED = False
_C.H.DEFLECT_DATA.FACETS.CANTING.ALGORITHM = 'standard'

_C.H.NURBS = CN()
_C.H.NURBS.MAX_ABS_NOISE = 0.01

_C.H.NURBS.SPLINE_DEGREE = 3
# Width, height, rows, cols (discretization dimensions), and
# facet/canting parameters given by `_C.H.IDEAL`.
# These are again the NURBS rows/cols of the control point matrix.
_C.H.NURBS.ROWS = 8
_C.H.NURBS.COLS = 8

_C.H.OTHER                              = CN()
_C.H.OTHER.FILENAME                     = 'tinker.obj'
_C.H.OTHER.USE_WEIGHTED_AVG             = True

# See `cfg.H.IDEAL.FACETS` for documentation.
_C.H.OTHER.FACETS = CN()
_C.H.OTHER.FACETS.POSITIONS = [0.0, 0.0, 0.0]
_C.H.OTHER.FACETS.SPANS_X = [0.0, float('inf'), 0.0]
_C.H.OTHER.FACETS.SPANS_Y = [-float('inf'), 0.0, 0.0]
_C.H.OTHER.FACETS.CANTING = CN()
_C.H.OTHER.FACETS.CANTING.ENABLED = False
_C.H.OTHER.FACETS.CANTING.ALGORITHM = 'standard'

# TODO add heliostat up vec ("rotation")

# AC = Ambiant Conditions
_C.AC                                   = CN()
_C.AC.RECEIVER                          = CN()
# in m in global coordinates

_C.AC.RECEIVER.CENTER                   = [0, -10, 0]
_C.AC.RECEIVER.PLANE_NORMAL             = [0, 1, 0] # NWU
_C.AC.RECEIVER.PLANE_X                  = 5. # in m
_C.AC.RECEIVER.PLANE_Y                  = 5. # in m
# These X and Y are height and width respectively.
_C.AC.RECEIVER.RESOLUTION_X             = 512
_C.AC.RECEIVER.RESOLUTION_Y             = 512

_C.AC.SUN                               = CN()

_C.AC.SUN.GENERATE_N_RAYS               = 1500
_C.AC.SUN.DISTRIBUTION                  = "Normal"                              #SWITCH FOR SOLAR DISTRIBUSTION: Normal, Point, Pillbox (not completly implemented)
_C.AC.SUN.REDRAW_RANDOM_VARIABLES       = False #TODO schauen wo das aufgerufen wird
_C.AC.SUN.NORMAL_DIST                   = CN()
_C.AC.SUN.NORMAL_DIST.MEAN              = [0,0]
_C.AC.SUN.NORMAL_DIST.COV               = [[0.002090**2, 0], [0, 0.002090**2]]



_C.TRAIN                                = CN()
_C.TRAIN.EPOCHS                         = 2500
_C.TRAIN.SUN_DIRECTIONS                 = CN()
_C.TRAIN.SUN_DIRECTIONS.CASE            ="random"   #SWITCH FOR SUN_DIRECTIONS DIRECTION VEKTOR GENERATION: vecs, random, grid

_C.TRAIN.SUN_DIRECTIONS.VECS            = CN()
_C.TRAIN.SUN_DIRECTIONS.VECS.DIRECTIONS = [[-0.43719268,  0.7004466,   0.564125  ],]

_C.TRAIN.SUN_DIRECTIONS.RAND            = CN()
_C.TRAIN.SUN_DIRECTIONS.RAND.NUM_SAMPLES = 10
_C.TRAIN.SUN_DIRECTIONS.RAND.LATITUDE    = 50.92 
_C.TRAIN.SUN_DIRECTIONS.RAND.LONGITUDE   = 6.36

_C.TRAIN.SUN_DIRECTIONS.GRID            = CN()
_C.TRAIN.SUN_DIRECTIONS.GRID.AZI_RANGE  = [-90, 90, 3] #Start,Stop,Step
_C.TRAIN.SUN_DIRECTIONS.GRID.ELE_RANGE  = [ 20, 80, 3] #Start,Stop,Step





_C.TRAIN.SCHEDULER                      = CN()
_C.TRAIN.SCHEDULER.NAME                 = "Exponential"                      #SWITCH FOR SCHEDULER: ReduceOnPLateau, Cyclic, OneCycle

_C.TRAIN.SCHEDULER.EXP          = CN()
_C.TRAIN.SCHEDULER.EXP.GAMMA    = 0.99

_C.TRAIN.SCHEDULER.ROP                  = CN()
_C.TRAIN.SCHEDULER.ROP.FACTOR           = 0.1
_C.TRAIN.SCHEDULER.ROP.MIN_LR           = 1e-7
_C.TRAIN.SCHEDULER.ROP.PATIENCE         = 20
_C.TRAIN.SCHEDULER.ROP.COOLDOWN         = 10
_C.TRAIN.SCHEDULER.ROP.VERBOSE          = True

_C.TRAIN.SCHEDULER.CYCLIC               = CN()
_C.TRAIN.SCHEDULER.CYCLIC.BASE_LR       = 1e-7
_C.TRAIN.SCHEDULER.CYCLIC.MAX_LR        = 8e-6
_C.TRAIN.SCHEDULER.CYCLIC.STEP_SIZE_UP  = 100
_C.TRAIN.SCHEDULER.CYCLIC.CYCLE_MOMENTUM= False
_C.TRAIN.SCHEDULER.CYCLIC.MODE          ="triangular2"

_C.TRAIN.SCHEDULER.ONE_CYCLE            = CN()
_C.TRAIN.SCHEDULER.ONE_CYCLE.MAX_LR     = 1e-4
_C.TRAIN.SCHEDULER.ONE_CYCLE.START_LR   = 1e-10
_C.TRAIN.SCHEDULER.ONE_CYCLE.FINAL_LR   = 1e-8
_C.TRAIN.SCHEDULER.ONE_CYCLE.PCT_START  = 0.3
_C.TRAIN.SCHEDULER.ONE_CYCLE.THREE_PHASE= True


_C.TRAIN.OPTIMIZER                      = CN()
_C.TRAIN.OPTIMIZER.NAME                 = "Adam"                              #SWITCH FOR OPTIMIZER: Adam, Adamax, AdamW
_C.TRAIN.OPTIMIZER.LR                   = 1e-4

_C.TRAIN.OPTIMIZER.BETAS                = [0.9, 0.999]
_C.TRAIN.OPTIMIZER.EPS                  = 1e-8
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY         = 0.91

_C.TRAIN.LOSS                           = CN()
_C.TRAIN.LOSS.NAME                      = "L1"                                  #SWITCH FOR LOSS: L1, MSE
_C.TRAIN.LOSS.USE_L1_WEIGHT_DECAY       = True
_C.TRAIN.LOSS.WEIGHT_DECAY_FACTOR       = 0.2


_C.TEST = CN()
_C.TEST.INTERVAL                        = 100

# Reduces test image array to 5, images will be generated with complete array

_C.TEST.SUN_DIRECTIONS                  = CN()
_C.TEST.SUN_DIRECTIONS.CASE             ="random"   #SWITCH FOR SUN DIRECTION VEKTOR GENERATION: vecs, random, grid

_C.TEST.SUN_DIRECTIONS.VECS             = CN()
_C.TEST.SUN_DIRECTIONS.VECS.DIRECTIONS  = [[-0.8662,  0.4890,  0.1026],] #Measurement Date 28.10.21 15:30


_C.TEST.SUN_DIRECTIONS.RAND             = CN()
_C.TEST.SUN_DIRECTIONS.RAND.NUM_SAMPLES = 5
_C.TEST.SUN_DIRECTIONS.RAND.LATITUDE    = 50.92 
_C.TEST.SUN_DIRECTIONS.RAND.LONGITUDE   = 6.36

_C.TEST.SUN_DIRECTIONS.GRID             = CN()
_C.TEST.SUN_DIRECTIONS.GRID.AZI_RANGE   = [-90, 90, 7] #Start,Stop,Step
_C.TEST.SUN_DIRECTIONS.GRID.ELE_RANGE   = [ 20, 80, 3] #Start,Stop,Step
_C.TEST.SUN_DIRECTIONS.GRID.PLOT        = True

_C.TEST.SUN_DIRECTIONS.SPHERIC          = CN()
_C.TEST.SUN_DIRECTIONS.SPHERIC.NUM_SAMPLES =10

_C.TEST.PLOT                            = CN()
_C.TEST.PLOT.SPHERIC                     =False
_C.TEST.PLOT.GRID                        =False
_C.TEST.PLOT.SEASON                      =False



def get_cfg_defaults():
    return _C.clone()


def load_config_file(cfg, config_file_loc):
    if len(os.path.splitext(config_file_loc)[1]) == 0:
        config_file_loc += '.yaml'
    cfg.merge_from_file(config_file_loc)
    
    # if experiment_name:
    #     cfg.merge_from_list(["ID", experiment_name])

    return cfg
