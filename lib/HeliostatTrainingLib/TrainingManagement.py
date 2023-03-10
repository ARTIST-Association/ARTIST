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
import zmq
import json
import time

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



class TrainingMessage:

    msg_type : typing.Optional[str] = None
    msg_id_key : str = 'msg_id'
    msg_type_key : str = 'msg_type'
    msg_count : int = 0

    def __init__(self, 
                 msg_dict : typing.Optional[typing.Union[str, typing.Dict[str, typing.Any]]] = None):

        if msg_dict and isinstance(msg_dict, str):
            self.msg_dict = json.loads(msg_dict)
        elif msg_dict:
            self.msg_dict = msg_dict
        else:
            self.msg_dict = {}

        if not self.msg_id_key in self.msg_dict:
            self.msg_dict[self.msg_id_key] = TrainingMessage.msg_count
            TrainingMessage.msg_count += 1

        if not self.msg_type_key in self.msg_dict:
            self.msg_dict[self.msg_type_key] = self.msg_type

    def msgType(self):
        return self.msg_dict[self.msg_type_key]

    def msgID(self):
        return self.msg_dict[self.msg_id_key]

    def toJSON(self):
        return json.dumps(self.msg_dict)

class AcknowledgementMessage(TrainingMessage):
     msg_type : typing.Optional[str] = 'Acknowledgement'
     acknowledged_id_key : str = 'acknowledged_id'

     def __init__(self, 
                  acknowledged_id : typing.Optional[int] = None,
                  msg_dict : typing.Optional[typing.Union[str, typing.Dict[str, typing.Any]]] = None):

        super().__init__(msg_dict=msg_dict)

        if isinstance(acknowledged_id, int):
            self.msg_dict[self.acknowledged_id_key] = acknowledged_id

     def acknowledgedID(self) -> int:
        return self.msg_dict[self.acknowledged_id_key]

class ScheduleTrainingMessage(TrainingMessage):
     
     msg_type : typing.Optional[str] = 'ScheduleTraining'
     training_dir_path_key : str = 'training_dir_path'

     def __init__(self, 
                  training_dir_path : typing.Optional[str] = None,
                  msg_dict : typing.Optional[typing.Union[str, typing.Dict[str, typing.Any]]] = None):

        super().__init__(msg_dict=msg_dict)

        if training_dir_path:
            self.msg_dict[self.training_dir_path_key] = 'training_dir_path'

     def trainingDirPath(self):
        return self.msg_dict[self.training_dir_path_key]

class TrainingRequestMessage(TrainingMessage):
    msg_type : typing.Optional[str] = 'TrainingRequest'
    training_dir_path_key : str = 'training_dir_path'

    def __init__(self, 
                  msg_dict : typing.Optional[typing.Union[str, typing.Dict[str, typing.Any]]] = None):

        super().__init__(msg_dict=msg_dict)

    def addTrainingDir(self, training_dir: typing.Optional[str]):
        if training_dir:
            self.msg_dict[self.training_dir_path_key] = training_dir

    def trainingDir(self) -> typing.Optional[str]:
        if self.training_dir_path_key in self.msg_dict:
            return self.msg_dict[self.training_dir_path_key]
        else:
            return None

class TrainingManagementServer:

    def __init__(self,
                 schedule_path : typing.Optional[str] = 'TrainingSchedule.csv',
                #  output_dir : typing.Optional[str] = None,
                 socket_id : str = "tcp://*:5555",
                #  schedule_length : int = 2000,
                 wait_time_secs : int = 10,
                ):
        self.schedule_path = schedule_path
        self.wait_time_secs = wait_time_secs
        self.socket_id = socket_id
        # self.schedule_length = schedule_length
        # self.output_dir = output_dir

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.socket_id)
        print('Training Management Active')
        # last_time = datetime.datetime.now()
        while True:
            message = socket.recv_string()
            if isinstance(message, str):
                self.readMessage(message=message, socket=socket)

    def readMessage(self, message: str, socket):
        try:
            m = TrainingMessage(msg_dict=message)
        except:
            print('Received Invalid Message!')

        received_type = m.msgType()
        if not received_type:
            print('Received Unkown Message Type!')

        elif received_type == ScheduleTrainingMessage.msg_type:
            print('Received Training Scheduling Request')
            m = ScheduleTrainingMessage(msg_dict=message)
            self.scheduleTraining(training_dir=m.trainingDirPath())
            m2 = AcknowledgementMessage(acknowledged_id=m.msgID())
            socket.send_string(m2.toJSON())

        elif received_type == TrainingRequestMessage.msg_type:
            print('Received Training Request')
            m = TrainingRequestMessage(msg_dict=message)
            training_dir = self.returnScheduledTraining()
            m.addTrainingDir(training_dir=training_dir)
            socket.send_string(m.toJSON())

    def scheduleTraining(self, training_dir):
        csv_tool = TrainingScheduleCSV()
        csv_tool.readCSV(csv_path=self.schedule_path)
        csv_tool.addDataRow(training_name=training_dir, start_date=None, completion_date=None)
        csv_tool.writeCSV(csv_path=self.schedule_path)

    def returnScheduledTraining(self) -> typing.Optional[str]:
        csv_tool = TrainingScheduleCSV()
        csv_tool.readCSV(csv_path=self.schedule_path)
        for i, (training_dir, start_time, completion_time) in enumerate(zip(csv_tool.names(), csv_tool.startDates(), csv_tool.completionDates())):
            if not start_time:
                csv_tool._data[csv_tool._start_date] = datetime.datetime.now()
                csv_tool.writeCSV(csv_path=self.schedule_path)
                return training_dir

class TrainingSchedulerClient:

    def __init__(self,
                 socket_id : str = "tcp://localhost:5555",
                ):
        self.socket_id = socket_id

    def scheduleTraining(self, training_dir : str):
        
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(self.socket_id)

        print('TrainingScheduler: Sending Scheduling Request')
        m_send = ScheduleTrainingMessage(training_dir_path=training_dir)
        socket.send_string(m_send.toJSON())
        print('TrainingScheduler: Awaiting Acknowledgement')

        while True:
            message = socket.recv_string()
            m_recv = TrainingMessage(msg_dict=message)
            if m_recv.msgType() == AcknowledgementMessage.msg_type:
                m_recv = AcknowledgementMessage(msg_dict=message)
                if m_recv.acknowledgedID() == m_send.msgID():
                    print('TrainingScheduler: Success')
                    break

class TrainingRunnerClient:

    def __init__(self,
                 socket_id : str = "tcp://localhost:5555",
                ):
        self.socket_id = socket_id

    def runTraining(self, loop : bool = False):
        
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(self.socket_id)

        count = 0
        while count == 0 or loop:
            count += 1
            print('TrainingRunner: Requesting Training')
            m_send = TrainingRequestMessage()
            socket.send_string(m_send.toJSON())
            print('TrainingRunner: Awaiting Acknowledgement')

            while True:
                message = socket.recv_string()
                m_recv = TrainingMessage(msg_dict=message)
                if m_recv.msgType() == TrainingRequestMessage.msg_type and m_recv.msgID() == m_send.msgID():
                    m_recv = TrainingRequestMessage(msg_dict=message)
                    # print('TrainingScheduler: Success')
                    training_dir = m_recv.trainingDir()
                    if not training_dir:
                        print('No Scheduled Training')
                    
                    break


if __name__ == '__main__':
    # tms = TrainingManagementServer()
    # tms.run()
    trc = TrainingRunnerClient()
    trc.runTraining()