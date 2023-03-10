import torch
import typing
from yacs.config import CfgNode
import sys
import os
from contextlib import redirect_stdout
import datetime
import json

# local dependencies
module_dir = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(module_dir)
from HeliostatDataset import HeliostatDataset
from HeliostatDatasetAnalyzer import HeliostatDatasetAnalyzer

lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(lib_dir)
from HeliostatKinematicLib.AlignmentModel import AbstractAlignmentModel
from HeliostatKinematicLib.AlignmentModelAnalyzer import AlignmentModelAnalyzer
from HeliostatKinematicLib.AlignmentModelBuilder import AlignmendModelBuilder

class HeliostatTraining:

    class TrainingResultKeys(typing.NamedTuple):
        train_deviation : str = 'Train Deviation'
        test_deviation : str = 'Test Deviation'
        eval_deviation : str = 'Eval Deviation'
        num_training : str = 'Num Training'
        num_testing : str = 'Num Testing'
        num_evaluation : str = 'Num Evaluation'
        test_distance : str = 'Testing Distance'
        eval_distance : str = 'Evaluation Distance'
        best_epoch : str = 'Best Epoch'
        max_epoch : str = 'Max Epoch'
    training_results_keys = TrainingResultKeys()

    json_indent = 4

    def __init__(self,
                 alignment_model: typing.Optional[AbstractAlignmentModel] = None,
                 dataset: typing.Optional[HeliostatDataset] = None, 
                 name : typing.Optional[str] = None,                
                 training_cfg: typing.Optional[CfgNode] = None,
                 dtype : torch.dtype = torch.get_default_dtype(),
                 device: torch.device = torch.device('cpu'),
                ):
        self._dataset = dataset
        self._alignment_model = alignment_model
        self._cfg = training_cfg if training_cfg else self.trainingConfig()
        self._name = name
        self._create_time = datetime.datetime.now()
        self._dtype = dtype
        self._device = device

        self._training_results = None

    def isReady(self) -> bool:
        if not self._dataset: return False
        if not self._alignment_model: return False
        return True

    def trainingDataDir(self, par_dir: str) -> str:
        time_str = self._create_time.strftime("%d_%m_%Y_%H_%M")
        training_data_dir = os.path.join(par_dir,self._name, time_str)
        count = 0
        while os.path.exists(training_data_dir):
            training_data_dir = os.path.join(par_dir, self._name, time_str + '_' + str(count))
            count = count + 1
        return training_data_dir

    def resultsDir(self, training_dir: str, restults_type: str) -> str:
        results_dir = os.path.join(training_dir, "Results", restults_type)
        return results_dir

    def trainingCfgPath(self, training_dir: str) -> str:
        cfg_path = os.path.join(training_dir, self._name + ".yaml")
        return cfg_path

    def modelSummaryPath(self, training_dir: str) -> str:
        summary_path = os.path.join(training_dir, 'alignment_model_summary.txt')
        return summary_path

    def datasetSummaryPath(self, training_dir: str) -> str:
        summary_path = os.path.join(training_dir, 'dataset_summary.txt')
        return summary_path

    def addTrainingResults(self, 
                           train_deviation: torch.Tensor, 
                           test_deviation : torch.Tensor,
                           eval_deviation : torch.Tensor,
                           num_training : int,
                           num_testing : int,
                           num_evaluation : int,
                           test_distance : int,
                           eval_distance : int,
                           best_epoch: int,
                           max_epoch: int,
                           ):
        if not self._training_results:
            self._training_results = {}

        self._training_results[self.training_results_keys.train_deviation] = train_deviation.item()
        self._training_results[self.training_results_keys.test_deviation] = test_deviation.item()
        self._training_results[self.training_results_keys.eval_deviation] = eval_deviation.item()
        self._training_results[self.training_results_keys.num_training] = num_training
        self._training_results[self.training_results_keys.num_testing] = num_testing
        self._training_results[self.training_results_keys.num_evaluation] = num_evaluation
        self._training_results[self.training_results_keys.test_distance] = test_distance
        self._training_results[self.training_results_keys.eval_distance] = eval_distance
        self._training_results[self.training_results_keys.best_epoch] = best_epoch
        self._training_results[self.training_results_keys.max_epoch] = max_epoch

    def saveResults(self, 
                    training_dir:str, 
                    results_type: str,
                    create_plots: bool = True,
                    fixate_disturbances: bool = False,
                    ):
        # create directories
        results_dir = self.resultsDir(training_dir=training_dir, restults_type=results_type)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # store results
        results_path = os.path.join(results_dir, 'training_results.json')
        with open(results_path, "w") as file:
            file.write(json.dumps(self._training_results))

        if create_plots:
            # plot errors
            dataset_analyzer = HeliostatDatasetAnalyzer(dataset=self._dataset)  
            dataset_analyzer.plotDataDistributionOverAxes(plot_errors=True,
                                                            show_plot=False,
                                                            save_path=os.path.join(results_dir, 'error_over_axes.png')
                                                            )
            dataset_analyzer.plotDataDistributionOverAngles(plot_errors=True,
                                                            show_plot=False,
                                                            save_path=os.path.join(results_dir, 'error_over_angles.png')
                                                            )
            dataset_analyzer.plotDataDistributionOverDates(plot_errors=True,
                                                            show_plot=False,
                                                            save_path=os.path.join(results_dir, 'error_over_time.png')
                                                            ) 
            dataset_analyzer.plotDataDistributionOverHausdorff(show_plot=False,
                                                            save_path=os.path.join(results_dir, 'error_over_hd.png')
                                                            )

            # alignment_analyzer = AlignmentModelAnalyzer(alignment_model=self._alignment_model, dtype=self._dtype, device=self._device)
            # alignment_analyzer.disturbancesByActuatorSteps(disturbance_list = self._alignment_model._disturbance_dict.keys(), 
            #                                                 save_analysis_path=os.path.join(results_dir, 'disturbance_parameters.png'),
            #                                                 show_plot=False,
            #                                                 )
            # alignment_analyzer.actuatorStepsBySolarAngles(aimpoint=torch.tensor([0,0,125], dtype=self._dtype, device=self._device),
            #                                             show_plot=False,
            #                                             save_analysis_path=os.path.join(results_dir, 'axes_steps.png'),
            #                                                 )

        if fixate_disturbances:
            self._alignment_model.fixateDisturbances()

        # save alignment
        self.toDirectory(par_dir=results_dir, save_config=False, create_plots=False, timed_dir=False)

    def fromDirectory(self, training_dir: str):
        self._name = os.path.basename(os.path.abspath(os.path.join(training_dir, os.pardir)))
        
        alignment_model_builder = AlignmendModelBuilder(dtype=self._dtype, device=self._device)
        alignment_model_dict = alignment_model_builder.loadAligmentModelDictFromJSON(json_path=os.path.join(training_dir, 'alignment_model.json'))
        self._alignment_model = alignment_model_builder.alignmentModelFromDict(alignment_model_dict=alignment_model_dict)
        
        # self._alignment_model = self._alignment_model.loadAlignmentModel(dir_path=training_dir)
        # self._dataset = self._dataset.loadDataset(dir_path=training_dir)
        dataset_config = {}
        with open(os.path.join(training_dir, HeliostatDataset.default_config_name), 'r') as file:
            dataset_config = json.load(file)

        self._dataset = HeliostatDataset(data_points = os.path.join(training_dir, HeliostatDataset.default_data_name), dataset_config=dataset_config, device=self._device)

        cfg_path = self.trainingCfgPath(training_dir=training_dir)
        self._cfg.defrost()
        self._cfg.merge_from_file(cfg_path)
        self._cfg.freeze()

        return self

    def toDirectory(self, par_dir: str, save_config: bool = True, create_plots: bool = True, timed_dir: bool = True) -> str:
        # create directories
        training_data_dir = self.trainingDataDir(par_dir=par_dir) if timed_dir else par_dir
        if not os.path.exists(training_data_dir):
            os.makedirs(training_data_dir)

        # save alignment
        # self._alignment_model.saveAlignmentModel(dir_path=training_data_dir)
        alignment_model_builder = AlignmendModelBuilder(dtype=self._dtype, device=self._device)
        alignment_model_builder.saveAlignmentModelDictToJSON(alignment_model=self._alignment_model, save_path=os.path.join(training_data_dir, 'alignment_model.json'))

        # save dataset
        self._dataset.toDirectory(output_dir = training_data_dir)

        # save training config
        if save_config:
            cfg_path = self.trainingCfgPath(training_dir=training_data_dir)
            with open(cfg_path, 'w') as f:
                with redirect_stdout(f): print(self._cfg.dump())

        # summary_path = self.modelSummaryPath(training_dir=training_data_dir)
        # with open(summary_path, 'w') as f:
        #     with redirect_stdout(f): print(self._alignment_model.summary())

        # summary_path = self.datasetSummaryPath(training_dir=training_data_dir)
        # with open(summary_path, 'w') as f:
        #     with redirect_stdout(f): print(self._dataset.summary())

        # analyze dataset
        if create_plots:
            dataset_analyzer = HeliostatDatasetAnalyzer(dataset=self._dataset)
            dataset_analyzer.plotDataDistributionOverAxes(plot_hausdorff=True, 
                                                        plot_epsilon_regions=True,
                                                        show_plot=False,
                                                        save_path=os.path.join(training_data_dir, 'hd_over_axes.png')
                                                        )
            dataset_analyzer.plotDataDistributionOverAngles(plot_hausdorff=True, 
                                                        plot_epsilon_regions=True,
                                                        show_plot=False,
                                                        save_path=os.path.join(training_data_dir, 'hd_over_angles.png')
                                                        )
            dataset_analyzer.plotDataDistributionOverDates(plot_hausdorff=True, 
                                                        show_plot=False,
                                                        save_path=os.path.join(training_data_dir, 'hd_over_time.png')
                                                        )
            # dataset_analyzer.plotHausdorffGauss(show_plot=False,
            #                                     save_path=os.path.join(training_data_dir, 'hd_gauss.png')
            #                                     )                                           

        # return training dir
        return training_data_dir

    def trainingConfig(
                     self,

                     # learning_rate_decay
                     learning_rate: float = 0.1,
                     lr_exp_decay: typing.Optional[float] = 0.997,

                     # optimizer
                     optimizer: str = 'Adam',

                     # weight_decay
                     weight_decay: float = 0.0,

                     # batches
                     batch_size: typing.Optional[float] = None,

                     # early stopping
                     early_stopping_patience: typing.Optional[int] = 200, 
                     early_stopping_min_delta: float = 0.01,

                     # optimizer
                     betas: typing.List[float] = [0.8, 0.9],
                     eps: float = 1.0e-08,

                     # training
                     epochs: int = 5000,
                     use_images: bool = False,
                     loss: str = 'MSE',
                    ) -> CfgNode:
            cfg = CfgNode()
            cfg.defrost()
            # cfg.set_new_allowed(True)
            cfg.BATCHES = CfgNode()
            cfg.EARLY_STOPPING = CfgNode()
            cfg.OPTIMIZER = CfgNode()
            cfg.OPTIMIZER.WEIGHT_DECAY = CfgNode()
            cfg.SCHEDULER = CfgNode()
            cfg.SCHEDULER.EXP = CfgNode()
            cfg.LOSS = CfgNode()
            

            # training
            cfg.EPOCHS = epochs
            cfg.USE_IMAGES = use_images
            cfg.LOSS.NAME = loss
            
            # learning rate
            cfg.OPTIMIZER.LR = learning_rate
            cfg.SCHEDULER.EXP.GAMMA = lr_exp_decay

            # weight decay
            cfg.OPTIMIZER.WEIGHT_DECAY.TOGGLE = True if weight_decay > 0.0 else False
            cfg.OPTIMIZER.WEIGHT_DECAY.TYPE = 'L1'
            cfg.OPTIMIZER.WEIGHT_DECAY.FACTOR = weight_decay

            # optimizer
            cfg.OPTIMIZER.NAME = optimizer
            cfg.OPTIMIZER.BETAS = betas
            cfg.OPTIMIZER.EPS = eps

            # batches
            cfg.BATCHES.TOGGLE = True if batch_size else False
            cfg.BATCHES.SIZE = batch_size

            # early stopping
            cfg.EARLY_STOPPING.TOGGLE = True if early_stopping_patience else False
            cfg.EARLY_STOPPING.PATIENCE = early_stopping_patience
            cfg.EARLY_STOPPING.MIN_DELTA = early_stopping_min_delta
            # cfg.set_new_allowed(False)
            cfg.freeze()

            return cfg