# system dependencies
import torch
import typing
from yacs.config import CfgNode
import sys
import os
from contextlib import redirect_stdout

# local dependencies
module_dir = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(module_dir)


lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(lib_dir)
from HeliostatTraining import HeliostatTraining
from HeliostatTraining.AlignmentTrainer import AbstractAlignmentTrainer

artist_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))
sys.path.append(artist_dir)
from main import mainFromConfig
from defaults import get_cfg_defaults

class ARTISTAlignmentTrainer(AbstractAlignmentTrainer):

    def __init__(self,
                 cfg: typing.Optional[CfgNode] = None,
                ):
        training_type = "ARTIST"

        self._cfg = cfg if cfg else self.defaultCfg()

        super().__init__(training_type=training_type)

    def createTrainingConfig(self, 
                             training: HeliostatTraining, 
                             save_path: typing.Optional[str] = None) -> CfgNode:

        if save_path:
            # save config
            with open(save_path, 'w') as f:
                with redirect_stdout(f): print(self._cfg.dump())

    def runTraining(self, 
                     training: HeliostatTraining,
                    ) -> HeliostatTraining :
         training = super().runTraining(training=training)

         # create ARTIST cfg
         cfg = self._cfg
         cfg.defrost()
         # cfg.set_new_allowed(True)
         train_cfg = CfgNode()
         train_cfg.TRAIN = training._cfg
         train_cfg.EXPERIMENT_NAME = training._name
         train_cfg.ID = training._name
         cfg.merge_from_other_cfg(train_cfg)
         # print(cfg.dump())
         # cfg.set_new_allowed(False)
         cfg.freeze()
         train_train_pe, train_eval_pe, test_test_pe, test_eval_pe, best_training, max_epoch = mainFromConfig(TRAINING=training, cfg=cfg)
         # training now includes:
         # - updated alignment and disturbance model
         # - alignment errors
         return best_training

    def defaultCfg(self) -> CfgNode:
        default_cfg = get_cfg_defaults()
        cfg = CfgNode()
        cfg.AC = CfgNode()
        cfg.AC.RECEIVER = CfgNode()
        cfg.AC.SUN = CfgNode()
        cfg.AC.SUN.NORMAL_DIST = CfgNode()
        cfg.H = CfgNode()
        cfg.H.DEFLECT_DATA = CfgNode()
        cfg.H.DEFLECT_DATA.FACETS = CfgNode()
        cfg.H.DEFLECT_DATA.FACETS.CANTING = CfgNode()
        cfg.TRAIN = CfgNode()
        cfg.TRAIN.LOSS = CfgNode()
        cfg.TRAIN.LOSS.MISS = CfgNode()
        
        # receiver
        cfg.AC.RECEIVER.PLANE_X = 10
        cfg.AC.RECEIVER.PLANE_Y = 10
        cfg.AC.RECEIVER.RESOLUTION_X = 10
        cfg.AC.RECEIVER.RESOLUTION_Y = 10

        # sun
        cfg.AC.SUN.DISTRIBUTION = 'Normal'
        cfg.AC.SUN.GENERATE_N_RAYS = 5
        cfg.AC.SUN.NORMAL_DIST.COV = [[4.3681e-06, 0], [0, 4.3681e-06]],
        cfg.AC.SUN.NORMAL_DIST.MEAN = [0, 0],
        cfg.AC.SUN.REDRAW_RANDOM_VARIABLES = False

        # heliostat
        cfg.H.SHAPE = 'real'
        cfg.H.DEFLECT_DATA.CONCENTRATORHEADER_STRUCT_FMT = '=5f2I2f'
        cfg.H.DEFLECT_DATA.FACETHEADER_STRUCT_FMT = '=i9fI'
        cfg.H.DEFLECT_DATA.FACETS.CANTING.ALGORITHM = 'standard'
        cfg.H.DEFLECT_DATA.FACETS.CANTING.FOCUS_POINT = 0.0
        cfg.H.DEFLECT_DATA.FILENAME = 'Helio_AA39_Rim0_STRAL-Input_211028212814.binp'
        cfg.H.DEFLECT_DATA.RAY_STRUCT_FMT = '=7f'
        cfg.H.DEFLECT_DATA.TAKE_N_VECTORS = 10
        cfg.H.DEFLECT_DATA.VERBOSE = True
        cfg.H.DEFLECT_DATA.ZS_PATH = 'Helio_AA39_Rim0_LocalResults_220303111914.csv'

        # training
        cfg.TRAIN.LOSS.MISS.NAME = 'L1'
        cfg.TRAIN.LOSS.MISS.FACTOR = 0.0
        cfg.TRAIN.LOSS.FACTOR = 0.0
        
        # other
        cfg.USE_GPU = False
        cfg.USE_NURBS = False
        cfg.USE_CACHE = False
        cfg.USE_FLOAT64 = True
        cfg.SAVE_RESULTS = True
        
        default_cfg.merge_from_other_cfg(cfg)
        
        return default_cfg