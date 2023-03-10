#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:56:07 2022

@author: user
"""
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import typing
import torch

class TrainingScheduleCSV:
    _sep = ';'
    _training_name = 'Training Name'
    _start_date = 'Start Date'
    _completion_date = 'Completion Date'
    _columns = [_training_name,
                _start_date,
                _completion_date, 
        ]
    
    def __init__(self,
                 dtype : torch.dtype = torch.get_default_dtype(),
                 device: torch.device = torch.device('cpu'),
                ):
        self._data = None
        self._dtype = dtype
        self._device = device
        
    def readCSV(self, csv_path):
        if os.path.isfile(csv_path):
            self._data = pd.read_csv(csv_path, sep = self._sep)
        else:
            self._data = pd.DataFrame(columns = self._columns)
            
        return self._data

    def names(self):
        return self._data[self._training_name]

    def startDates(self) -> typing.List[typing.Optional[datetime.datetime]]:
        data_list = []
        for d in self._data[self._start_date]:
            if not d or d == 'False' or d == 'None' or len(d) == 0:
                data_list.append(None)

            else:
                try:
                    d = d if len(d) <= 19 else d[:19]
                    data_list.append(datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S'))
                except Exception as e:
                    data_list.append(None)
                
        return data_list

    def completionDates(self) -> typing.List[typing.Optional[datetime.datetime]]:
        data_list = []
        for d in self._data[self._completion_date]:
            if not d or d == 'False' or d == 'None' or len(d) == 0:
                data_list.append(None)

            else:
                try:
                    d = d if len(d) <= 19 else d[:19]
                    data_list.append(datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S'))
                except Exception as e:
                    data_list.append(None)
                
        return data_list

    def addDataRow(self,
                   training_name: str,
                   start_date : typing.Optional[datetime.datetime],
                   completion_date : typing.Optional[datetime.datetime],
                   ):
        start_date = start_date if start_date else False
        completion_date = completion_date if completion_date else False
        data = [training_name,
                start_date,
                completion_date
                ]
        
        if self._data is None:
            self._data = pd.DataFrame(columns = self._columns)
            

        # print('new: ' + str(len(data)) + ' old: ' + str(self._data.shape))
        # print(data)
        
        self._data.loc[len(self._data.index)] = data
        # self._data.append(pd.Series(data, index = self._data.columns[:len(data)]), ignore_index=True)
    
    def writeCSV(self, csv_path):
        if isinstance(self._data, pd.DataFrame):
            self._data.to_csv(csv_path, sep=self._sep, index=False)