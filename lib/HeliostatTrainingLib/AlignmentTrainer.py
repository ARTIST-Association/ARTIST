#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:04:13 2023

@author: moritz.leibauer@synhelion.com
"""

# system dependencies
import torch
import typing
from yacs.config import CfgNode
import sys
import os
from contextlib import redirect_stdout
import datetime

# local dependencies
module_dir = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(module_dir)
from HeliostatDataset import HeliostatDataset
from HeliostatDatasetBuilder import HeliostatDatasetBuilder
from HeliostatTraining import HeliostatTraining
from HeliostatDataset import HeliostatDataPoint

lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(lib_dir)
from CSVToolsLib.TrainingScheduleCSV import TrainingScheduleCSV
from HeliostatKinematicLib.AlignmentModel import AbstractAlignmentModel, HeliokonAlignmentModel
from HeliostatKinematicLib.AlignmentModelBuilder import AlignmendModelBuilder
from HeliostatKinematicLib.AlignmentModelAnalyzer import AlignmentModelAnalyzer
from NeuralNetworksLib.EarlyStopping import EarlyStopper

class AbstractAlignmentTrainer:

    def __init__(self, 
                 training_type: str="Default",
                 dtype: torch.dtype = torch.get_default_dtype(),
                 device: torch.device = torch.device('cpu'),
                 ):
        self.training_type = training_type
        self._dtype = dtype
        self._device = device

        # abstract class guard
        if type(self).__name__ == AbstractAlignmentTrainer.__name__:
            raise Exception("Don't implement an abstract class!")

    def createTrainingConfig(self, 
                             training: HeliostatTraining, 
                             save_path: typing.Optional[str] = None) -> CfgNode:
        # abstract class guard
        raise Exception("Abstract method must be overridden!")

    def runTraining(self, 
                     training: HeliostatTraining,
                    ) -> HeliostatTraining:
         print('RUNNING training: ' + training._name)
         return training

class DUMMYAlignmentTrainer(AbstractAlignmentTrainer):

    def __init__(self):
        training_type = "DUMMY"
        super().__init__(training_type=training_type)

    def createTrainingConfig(self, 
                             training: HeliostatTraining, 
                             save_path: typing.Optional[str] = None) -> CfgNode:
        cfg = CfgNode()

        if save_path:
            # save config
            with open(save_path, 'w') as f:
                with redirect_stdout(f): print(cfg.dump())

    def runTraining(self, 
                     training: HeliostatTraining,
                     fixate_disturbances : bool = False,
                    ) -> HeliostatTraining :
         training = super().runTraining(training=training)
         return training

class SIMPLEAlignmentTrainer(AbstractAlignmentTrainer):

    def __init__(self,
                 dtype: torch.dtype = torch.get_default_dtype(),
                 device: torch.device = torch.device('cpu'),
                ):
        training_type = "SIMPLE"
        super().__init__(training_type=training_type, dtype=dtype, device=device)

    def createTrainingConfig(self, 
                             training: HeliostatTraining, 
                             save_path: typing.Optional[str] = None) -> CfgNode:
        cfg = CfgNode()

        if save_path:
            # save config
            with open(save_path, 'w') as f:
                with redirect_stdout(f): print(cfg.dump())

    def updateDataset(self, alignment_model: AbstractAlignmentModel, dataset: HeliostatDataset):
        # updated_dataset = dataset
        # print('0/' + str(len(dataset)))

        for key in dataset._training_data_points:
            dp = dataset._data_points[key]
            alignment_dict = alignment_model.alignFromDataPoint(data_point=dp)
            alignment_dict['Type'] = 'Train'
            dataset._data_points[key].training_results.update(alignment_dict)

        for key in dataset._testing_data_points:
            dp = dataset._data_points[key]
            alignment_dict = alignment_model.alignFromDataPoint(data_point=dp)
            alignment_dict['Type'] = 'Test'
            dataset._data_points[key].training_results.update(alignment_dict)

        for key in dataset._evaluation_data_points:
            dp = dataset._data_points[key]
            alignment_dict = alignment_model.alignFromDataPoint(data_point=dp)
            alignment_dict['Type'] = 'Eval'
            dataset._data_points[key].training_results.update(alignment_dict)

        # for i, (key, dp) in enumerate(dataset.items()):
        #     alignment_dict = alignment_model.alignFromDataPoint(data_point=dp)
        #     updated_dataset[key].training_results.update(alignment_dict)
            
        # return dataset

    def lossOOB(self, training: HeliostatTraining,) -> torch.Tensor:
        alignment_model_analyzer = AlignmentModelAnalyzer(alignment_model=training._alignment_model, dtype=self._dtype, device=self._device)
        
        return alignment_model_analyzer.checkOOB(aimpoint=torch.tensor([0,0,125], dtype=self._dtype, device=self._device),
                                                 angle_steps= [90, 30]
                                                )

    def runTraining(self, 
                     training: HeliostatTraining,
                     check_oob: bool = False,
                     disable_weight_decay: bool = False,
                    ) -> HeliostatTraining :
         training = super().runTraining(training=training)

         # optimizer setup
         training_params = training._alignment_model.getTrainingParams()
         print('FOUND Training Params:')
         print(training_params)
         print('')
         if isinstance(training_params, dict):
            training_params = training_params.values()
        
         #  optimizer = torch.optim.Adam(params=training_params, lr=training._cfg.OPTIMIZER.LR, betas=training._cfg.OPTIMIZER.BETAS, eps=training._cfg.OPTIMIZER.EPS)
         lr = training._cfg.OPTIMIZER.LR
         if training._cfg.OPTIMIZER.WEIGHT_DECAY.TOGGLE and not disable_weight_decay:
             optimizer = torch.optim.Adam(params=training_params, lr=lr, weight_decay=training._cfg.OPTIMIZER.WEIGHT_DECAY.FACTOR)
         else:
            optimizer = torch.optim.Adam(params=training_params, lr=lr)
         #  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.3, min_lr=0.001, patience=int(training._cfg.EARLY_STOPPING.PATIENCE /3 + 0.5))
         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, training._cfg.SCHEDULER.EXP.GAMMA)

         # early stopper setup
         early_stopper = EarlyStopper(patience=training._cfg.EARLY_STOPPING.PATIENCE, min_delta=training._cfg.EARLY_STOPPING.MIN_DELTA) if training._cfg.EARLY_STOPPING.TOGGLE else None

         # training
         best_test_epoch_data = None
         best_test_epoch_loss = None
         best_test_epoch = None
         alignment_model_builder = AlignmendModelBuilder(dtype=self._dtype, device=self._device)
         for epoch in range(training._cfg.EPOCHS):
            # end training if updating training_parameters fails
            for tp in training_params:
                if torch.isnan(tp):
                    break

            # reset optimizer
            optimizer.zero_grad()

            # update datasets
            self.updateDataset(alignment_model=training._alignment_model, dataset=training._dataset)

            # acs = torch.zeros(2)
            # print(training._alignment_model._disturbance_model.predictDisturbances(normalized_actuator_steps=acs))

            # epoch information
            epoch_str = '[' + str(epoch) + '/' + str(training._cfg.EPOCHS) + '] ->\t'
            epoch_str = epoch_str + 'Train: ' + str(training._dataset.avgTrainError().item()) + '\t'
            if len(training._dataset.testingDataset()) > 0:
                epoch_str = epoch_str + 'Test: ' + str(training._dataset.avgTestError().item()) + '\t'
            if len(training._dataset.evaluationDataset()) > 0:
                epoch_str = epoch_str + 'Eval: ' + str(training._dataset.avgEvalError().item()) + '\t'
            # print('\tNormal: ' + str(n.detach().numpy()) + '\tNormal_Target: ' + str(nt.detach().numpy()) + '\tPivoting Point: ' + str(pp.detach().numpy()))

            # loss
            loss = torch.pow(training._dataset.avgTrainError(), torch.tensor(2, dtype=self._dtype, device=self._device))
            if check_oob:
                loss += self.lossOOB(training=training)

            # weight decay
            if training._cfg.OPTIMIZER.WEIGHT_DECAY.TOGGLE and not disable_weight_decay:
                if training._cfg.OPTIMIZER.WEIGHT_DECAY.TYPE == 'L1':
    
                    training_params = training._alignment_model._disturbance_model._model.parameters()
                    # training_params = torch.flatten(training_params)
                    # l1_loss = torch.sum(training_params) * training._cfg.OPTIMIZER.WEIGHT_DECAY.FACTOR
                    if isinstance(training_params, dict):
                        training_params = training_params.values()

                    l1_loss = torch.tensor(0, dtype=self._dtype, device=self._device)
                    for p in training_params:
                        p_flat = torch.flatten(p)
                        if len(p_flat) > 1:
                            for pf in p_flat:
                                l1_loss += torch.abs(pf)
                        else:
                            l1_loss += torch.abs(p_flat)

                    # l1_loss = training_params
                    print('Model Weight: ' + str(l1_loss.item()))

                    # loss += l1_loss

            if torch.isnan(loss):
                break

            test_loss = training._dataset.avgTestError().tolist()
            test_loss_sq = torch.pow(test_loss, torch.tensor(2, dtype=self._dtype, device=self._device))
            if loss > 0  and (not best_test_epoch_loss or test_loss_sq < best_test_epoch_loss):
                best_test_epoch_loss = test_loss_sq
                best_test_epoch_data = alignment_model_builder.dictFromAlignmentModel(alignment_model=training._alignment_model)
                best_test_epoch = epoch

            # scheduljer.step(training._dataset.avgTrainError())
            scheduler.step()
            epoch_str = epoch_str + 'LR: ' + str(scheduler.optimizer.param_groups[0]['lr'])

            # early stopping
            if early_stopper:
                es_toggle, es_delta = early_stopper.checkEpoch(testing_loss=test_loss_sq, epoch=epoch)
                if not es_toggle:
                    print('Early Stopping Toggle At Epoch: ' + str(epoch))
                    break
                elif es_delta > 1:
                    epoch_str = epoch_str + '\t ES: ' + str(es_delta * 100.0 / training._cfg.EARLY_STOPPING.PATIENCE) + '%'

            print(epoch_str)

            try:
                loss.backward(retain_graph=True)
                optimizer.step()
            except:
                print('FAILED Computing Loss!')
                break

        #  orig_stdout = sys.stdout
        #  if os.path.exists('/Users/Synhelion/Desktop/FieldSimulationFixed/log.txt'):
        #     with open('/Users/Synhelion/Desktop/FieldSimulationFixed/log.txt', 'r') as file:
        #         epoch_str = file.read() + '\n' + epoch_str
        #  with open('/Users/Synhelion/Desktop/FieldSimulationFixed/log.txt', 'w') as file:
        #     sys.stdout = file
        #     print(epoch_str)
        #     sys.stdout = orig_stdout
        #     file.close()

         print('Restoring Model From Epoch: ' + str(best_test_epoch) + ' With Test: ' + str(best_test_epoch_loss))
         training._alignment_model = alignment_model_builder.alignmentModelFromDict(alignment_model_dict=best_test_epoch_data)
         self.updateDataset(alignment_model=training._alignment_model, dataset=training._dataset)
         training.addTrainingResults(train_deviation=training._dataset.avgTrainError(),
                                     test_deviation=training._dataset.avgTestError(),
                                     eval_deviation=training._dataset.avgEvalError(),
                                     num_training=len(training._dataset.trainingDataset()),
                                     num_testing=len(training._dataset.testingDataset()),
                                     num_evaluation=len(training._dataset.evaluationDataset()),
                                     test_distance = training._dataset.avgTestDistance().item(),
                                     eval_distance = training._dataset.avgEvalDistance().item(),
                                     max_epoch=epoch,
                                     best_epoch=best_test_epoch,
                                    )

         return training

class AlignmentTrainingManager:

    default_training_dir_name = 'HeliostatTraining'
    default_config_file_ext = '.yaml'
    training_results_dir = 'Results'
    default_training_schedule_name = 'TrainingSchedule.csv'

    def __init__(self,
                 trainer : typing.Optional[AbstractAlignmentTrainer] = None,
                 output_dir : typing.Optional[str] = None,
                 config_file_ext : typing.Optional[str] = None,
                 alignment_model_type = HeliokonAlignmentModel,
                #  dataset_type = HeliOSHeliostatDataset,
                 create_plots : bool = True,
                 dtype: torch.dtype = torch.get_default_dtype(),
                 device: torch.device = torch.device('cpu'),
                ):
        self._dtype = dtype
        self._device = device
            
        self._trainer = trainer
        self._output_dir = output_dir if output_dir else os.path.join(os.path.dirname(__file__), AlignmentTrainingManager.default_training_dir_name)
        self._config_file_ext = config_file_ext if config_file_ext else AlignmentTrainingManager.default_config_file_ext

        self._alignment_model_type = alignment_model_type
        # self._dataset_type = dataset_type

        self._create_plots = create_plots

        self._training_list = []
        self._scheduled_trainings = {}
        if os.path.exists(self._output_dir):
            self._training_list = [f.path for f in os.scandir(self._output_dir) if self._evaluateDirectory(dir_path = f)]

        print('Heliostat Alignment Trainer')
        print('- Training Schedule: ' + self.scheduleFilePath())
        # self.updateScheduledTrainings()

    def _evaluateDirectory(self, dir_path) -> bool:
        # directories only
        if dir_path.is_dir():
            return True
        
        return False

    def addTraining(self, 
                    training : HeliostatTraining,
                    ):

        # create directory if not existing
        training_dir_path = training.toDirectory(par_dir=self._output_dir, create_plots=self._create_plots)
        print('Added training: ' + training_dir_path)
        self._scheduled_trainings[training_dir_path] = (False, False)
        self.updateScheduledTrainings()

    def scheduleFilePath(self) -> str:
        schedule_file_path = os.path.join(self._output_dir, AlignmentTrainingManager.default_training_schedule_name)
        return schedule_file_path

    def updateScheduledTrainings(self) -> typing.List[str]:
        schedule_file_path = self.scheduleFilePath()
        new_scheduled_trainings = {}

        if os.path.exists(schedule_file_path):
            csv_tool = TrainingScheduleCSV()
            csv_tool.readCSV(csv_path=schedule_file_path)
            for training_dir, start_time, completion_time in zip(csv_tool.names(), csv_tool.startDates(), csv_tool.completionDates()):
                new_scheduled_trainings[training_dir] = (start_time, completion_time)

        for key, value in self._scheduled_trainings.items():
            if not key in new_scheduled_trainings:
                new_scheduled_trainings[key] = value
            elif not new_scheduled_trainings[key][0] and value[0]:
                new_scheduled_trainings[key] = value
            elif value[0] == new_scheduled_trainings[key][0] and value[1]:
                new_scheduled_trainings[key] = value

        self._scheduled_trainings = new_scheduled_trainings
        self.saveTrainingSchedule(self.scheduleFilePath())

        num_open_trainings = self.numOpenTrainings()
        print('Updated Scheduled Trainings: Open: ' + str(num_open_trainings) + ' Completed: ' + str(len(self._scheduled_trainings) - num_open_trainings))
        return num_open_trainings

    def numOpenTrainings(self) -> int:
        num_open_trainings = 0
        for sd, (tic, toc) in self._scheduled_trainings.items():
            if not isinstance(toc, datetime.datetime) and not isinstance(tic, datetime.datetime):
                num_open_trainings += 1
        return num_open_trainings

    def saveTrainingSchedule(self, file_path: str):
        num_open_trainings = self.numOpenTrainings()
        print('Updated Scheduled Trainings: Open: ' + str(num_open_trainings) + ' Completed: ' + str(len(self._scheduled_trainings) - num_open_trainings))

        csv_tool = TrainingScheduleCSV()
        for training_dir, sc_time in self._scheduled_trainings.items():
            csv_tool.addDataRow(training_name=training_dir, start_date=sc_time[0], completion_date=sc_time[1])
        csv_tool.writeCSV(csv_path=file_path)

    def loadTraining(self, training_dir_path):
        name = os.path.basename(training_dir_path)
        training_path = os.path.join(training_dir_path, name + self._config_file_ext)

    def runScheduledTrainings(self, 
                                include_completed : bool = False,
                                #fixate_disturbances : bool = False,
                                pre_alignment_model = None,
                        ):
        if len(self._scheduled_trainings) == 0:
            print("WARNING: No scheduled trainings!")
        elif not self._trainer:
            print('WARNING: Trainer not selected!')
            return

        fixated_alignment = None
        alignment_model_builder = AlignmendModelBuilder(dtype=self._dtype, device=self._device)
        
        loop_cond = True
        while loop_cond:
            if self.updateScheduledTrainings() > 0:
                for training_dir, (start_time, completion_time) in self._scheduled_trainings.items():
                    if os.path.exists(training_dir):
                        if (not completion_time and not start_time) or include_completed:
                            tic = datetime.datetime.now()
                            self._scheduled_trainings[training_dir] = (tic, None)
                            self.updateScheduledTrainings()
                            training = HeliostatTraining(#alignment_model=self._alignment_model_type(),
                                                        #  dataset=self._dataset_type(),
                                                        name='Training',
                                                        dtype=self._dtype,
                                                        device=self._device,
                                                            )
                            training = training.fromDirectory(training_dir=training_dir)
                            if pre_alignment_model:
                                training._alignment_model._disturbance_model = pre_alignment_model
                                training = self._trainer.runTraining(training=training, disable_weight_decay=True)

                                training._alignment_model.fixateDisturbances()
                                model_data = alignment_model_builder.dictFromAlignmentModel(alignment_model=training._alignment_model)
                            
                                training = training.fromDirectory(training_dir=training_dir)
                                distubance_model = training._alignment_model._disturbance_model
                                training._alignment_model = alignment_model_builder.alignmentModelFromDict(alignment_model_dict=model_data)
                                training._alignment_model._disturbance_model = distubance_model

                            # if fixated_alignment:
                            #     training._alignment_model = alignment_model_builder.alignmentModelFromDict(alignment_model_dict=fixated_alignment)
                            training = self._trainer.runTraining(training=training)
                            training.saveResults(training_dir=training_dir, results_type=self._trainer.training_type, create_plots=self._create_plots)
                            # if fixate_disturbances:
                            #         fixated_alignment = alignment_model_builder.dictFromAlignmentModel(alignment_model=training._alignment_model)
                            
                            toc = datetime.datetime.now()
                            self._scheduled_trainings[training_dir] = (tic, toc)
                            break
                    else:
                        print("WARNING: No such training: " + training_dir)
            else:
                loop_cond = False

# def TrainingSchedulingTest():
#     builder = HeliostatDatasetBuilder(dataset=HeliOSHeliostatDataset())
#     heliostat_list = ['AM.43']
#     file_path = '/Users/Synhelion/Downloads/calibdata.csv'
#     # dataset = builder.buildRandomlyFromPercentage(file_path=file_path,
#     #                                               num_train_data=100,
#     #                                               perc_test=0.5,
#     #                                               heliostats_list=heliostat_list
#     #                                                 )
#     dataset = builder.buildWithMaximizedHausdorffDistance(file_path=file_path,
#                                                         num_train_data=100,
#                                                         perc_test=0.5,
#                                                         heliostats_list=heliostat_list
#                                                             )
#     alignment = HeliokonAlignmentModel()
#     training_1 = HeliostatTraining(dataset=dataset, alignment_model=alignment, name='AM.43_100_Train_0.5')
#     trainer = DUMMYAlignmentTrainer()
#     training_manager = AlignmentTrainingManager(trainer=trainer)
#     training_manager.addTraining(training=training_1)
#     training_manager.runScheduledTrainings()

# if __name__ == '__main__':
#     TrainingSchedulingTest()