#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:04:13 2023

@author: moritz.leibauer@synhelion.com
"""

# system dependencies
import sys
import os
import typing
import torch
import datetime

# local dependencies
lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir, 'lib'))
sys.path.append(lib_dir)
from CSVToolsLib.HeliostatDatasetCSV import HeliOSDatasetCSV
import HeliostatTrainingLib.HeliostatDatasetBuilder as HDB
import HeliostatTrainingLib.HeliostatDataset as HD
import HeliostatTrainingLib.HeliostatTraining as HT
import HeliostatTrainingLib.AlignmentTrainer as AT
import HeliostatKinematicLib.AlignmentModel as AM
import HeliostatKinematicLib.AlignmentDisturbanceModel as ADM
import NeuralNetworksLib.InputEncoding as IE
from HeliostatTrainingLib.TrainingManagement import TrainingManagementServer
import zmq
import time
if __name__ == '__main__':
    tms = TrainingManagementServer()
    tms.run()