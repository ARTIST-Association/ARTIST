#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:04:13 2023

@author: user
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
from HeliostatDataset import AbstractHeliostatDataset, HeliOSHeliostatDataset, HeliostatDataPoint
from HeliostatDatasetBuilder import HeliostatDatasetBuilder
from HeliostatTraining import HeliostatTraining
import HeliostatTraining as HT
import AlignmentTrainer as AT

lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(lib_dir)
from HeliostatKinematicLib.AlignmentModel import AbstractAlignmentModel
import HeliostatKinematicLib.AlignmentModel as AM
import HeliostatKinematicLib.AlignmentDisturbanceModel as ADM
import NeuralNetworksLib.InputEncoding as IE


class FieldSimulator:

    def __init__(self, 
                 dataset_path: str,
                 alignment_model: AbstractAlignmentModel,
                 working_dir: typing.Optional[str] = None,
                 dtype: torch.dtype = torch.get_default_dtype(),
                 device: torch.device = torch.device('cpu')
                 ):
        self._alignment_model = alignment_model
        self._dataset_path = dataset_path
        self._working_dir = working_dir
        self._dtype = dtype
        self._device = device

    def scheduleDBDSimulation(self,
                                heliostat: str,
                                start_date: typing.Optional[datetime.datetime] = None,
                                end_date: typing.Optional[datetime.datetime] = None,
                                ):
        complete_dataset = HeliOSHeliostatDataset(dtype=self._dtype, device=self._device)
        complete_dataset.loadDataFromFile(file_path=self._dataset_path, 
                                          heliostat_list=[heliostat], 
                                          start_date=start_date,
                                          end_date=end_date,
                                          )

        training_manager = AT.AlignmentTrainingManager(output_dir = self._working_dir, dtype=self._dtype, device=self._device)

        dp_sorted_by_date = [dp for dp in sorted(complete_dataset._dataset, key= lambda item: item.created_at)]
        dataset_end_date = dp_sorted_by_date[-1].created_at
        for i, dp in enumerate(dp_sorted_by_date):
            dataset = complete_dataset
            train_end_date = dp.created_at

            if i >= len(dp_sorted_by_date) - 1:
                break

            dataset.setTrainingData(created_at_range=[start_date, train_end_date])

            test_end_date = dp_sorted_by_date[i+1].created_at
            dataset._testing_dataset = {dp_sorted_by_date[i+1]._unique_id: dp_sorted_by_date[i+1]}

            dataset.setEvaluationData(created_at_range=[test_end_date,dataset_end_date])
            if len(dataset._evaluation_dataset) == 0:
                dataset._evaluation_dataset = dataset._testing_dataset

            dataset._updateHaussdorfDistances()

            training = HT.HeliostatTraining(dataset=dataset, alignment_model=alignment, name=str(i), dtype=self._dtype, device=self._device)
            training_manager.addTraining(training)

    def runSimulation(self, fixate_disturbances: bool = False):
        trainer = AT.SIMPLEAlignmentTrainer(dtype=torch.float64)
        training_manager = AT.AlignmentTrainingManager(trainer=trainer, output_dir = self._working_dir, dtype=self._dtype, device=self._device)
        training_manager.runScheduledTrainings(fixate_disturbances=fixate_disturbances),

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    working_dir = '/Users/Synhelion/Desktop/FieldSimulationFixed'
    dataset_path = '/Volumes/UBUNTU 20_0/data/calibdata_BA28.csv'
    heliostat = 'BA.28'
    position = torch.tensor([-35.2, 171.4, 88.68751]) # BA.28

    # rigid body disturbances
    disturbance_list = [
        'position_azim',
        'position_elev',
        'position_rad',

        'joint_2_cosys_pivot_azim',
        'joint_2_cosys_pivot_elev',
        'joint_2_cosys_pivot_rad',

        'concentrator_cosys_pivot_azim',
        'concentrator_cosys_pivot_elev',
        'concentrator_cosys_pivot_rad',

        'joint_1_east_tilt',
        'joint_1_north_tilt',
        'joint_1_up_tilt',

        'joint_2_east_tilt',
        'joint_2_north_tilt',
        'joint_2_up_tilt',

        'concentrator_east_tilt',
        'concentrator_north_tilt',
        'concentrator_up_tilt',

        'actuator_1_increment',
        'actuator_2_increment',
    ]

    # nn disturbances
    # disturbance_list = [
    #     'concentrator_cosys_pivot_azim',
    #     'concentrator_cosys_pivot_elev',
    #     'concentrator_cosys_pivot_rad',

    #     'concentrator_east_tilt',
    #     'concentrator_north_tilt',
    #     'concentrator_up_tilt',
    # ]

    disturbance_model = ADM.RigidBodyAlignmentDisturbanceModel(disturbance_list = disturbance_list, dtype=torch.float64)
    # input_encoder = IE.PseudoFourierActuatorEncoder(num_actuators = 2,
    #                                                 )
    # disturbance_model = ADM.SNNAlignmentDisturbanceModel(disturbance_list=disturbance_list,
    #                                                      hidden_dim = 2,
    #                                                      n_layers = 3,
    #                                                      input_encoder = input_encoder,
    #                                                      dtype=torch.float64,
    #                                                     )
    
    alignment = AM.HeliokonAlignmentModel(position=position, disturbance_model=disturbance_model, dtype=torch.float64)


    start_date = datetime.datetime(year=2022, month=1, day=1)
    end_date = datetime.datetime(year=2022, month=6, day=20)

    fs = FieldSimulator(dataset_path=dataset_path,
                        alignment_model=alignment,
                        working_dir=working_dir,
                        dtype=torch.float64
                        )

    fs.scheduleDBDSimulation(heliostat = heliostat,
                             start_date = start_date,
                             end_date = end_date
                             )

    fs.runSimulation(fixate_disturbances=True)