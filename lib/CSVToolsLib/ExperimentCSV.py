#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:56:07 2022

@author: user
"""
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import torch

class ExperimentCSV:
    _sep = ';'
    
    _columns = ['Experiment ID',
                'Name', 'Kinematic',
                'Train Start', 'Train End', 'Num Train',
                'Test Start', 'Test End', 'Num Test',
                'Eval Start', 'Eval End', 'Num Eval',
                'Measurement Error', 'Seed',
                'ACC Train', 'ACC Test', 'ACC Eval',
                'Total Epochs',
                'Batch Size',
                'Optimizer', 'Alignment Loss', 
                'LR', 'LR Decay',
                'Weight Decay',
                'Experiment Type' 
        ]
    
    def __init__(self):
        self._data = None
        
    def readCSV(self, csv_path):
        if os.path.isfile(csv_path):
            self._data = pd.read_csv(csv_path, sep = self._sep)
        else:
            self._data = pd.DataFrame(columns = self._columns)
            
        return self._data

    def hs_names(self):
        return self._data['Name']

    def kinematic(self):
        return self._data['Kinematic']
    
    def num_points_train(self):
        return self._data['Num Train']

    def eval_pe(self):
        return [float(d[7:-1]) for d in self._data['ACC Eval']]

    def batch_size(self):
        return self._data['Batch Size']

    def optimizer(self):
        return self._data['Optimizer']

    def alignment_loss(self):
        return self._data['Alignment Loss']

    def lr(self):
        return self._data['LR']

    def weight_decay(self):
        return self._data['Weight Decay']

    def addDataRow(self,
                   experiment_id: int,
                   hs_name: str,
                   kinematic: str,
                   train_month_start: int, train_month_end: int, num_points_train: int,
                   test_month_start: int, test_month_end: int, num_points_test: int,
                   eval_month_start: int, eval_month_end: int, num_points_eval: int,
                   meas_error: float, seed: int, 
                   train_pe: float, test_pe: float, eval_pe: float,
                   max_epoch: int,
                   batch_size: int, 
                   optimizer: str, alignment_loss: str, 
                   lr: float, lr_exp_decay: float,
                   weight_decay: float,
                   experiment_type: str
                   ):
        data = [experiment_id,
                hs_name, kinematic,
                train_month_start, train_month_end, num_points_train,
                test_month_start, test_month_end, num_points_test,
                eval_month_start, eval_month_end, num_points_eval,
                meas_error, seed,
                train_pe, test_pe, eval_pe,
                max_epoch,
                batch_size,
                optimizer, alignment_loss, 
                lr, lr_exp_decay,
                weight_decay,
                experiment_type
                ]
        
        if self._data is None:
            self._data = pd.DataFrame(columns = self._columns)
            

        # print('new: ' + str(len(data)) + ' old: ' + str(self._data.shape))
        # print(data)
        
        self._data.loc[len(self._data.index)] = data
        # self._data.append(pd.Series(data, index = self._data.columns[:len(data)]), ignore_index=True)
    
    def writeCSV(self, csv_path):
        self._data.to_csv(csv_path, sep=self._sep, index=False)

    def plot(self):

        def shapeFromLoss(loss):
            shape_dict = {
                'MSE' : '^',
                'L1' : 'o',
            }
            
            shapes = []
            for l in loss:
                shapes.append(shape_dict[l])

            return shapes

        def colorFromKinematic(kinematic):
            kin_dict = {
                'Heliokon': 'coral',
                'HeliokonDense1': 'teal',
                'HeliokonDense2': 'lightskyblue',
            }
            colors = []
            for k in kinematic:
                colors.append(kin_dict[k])
            return colors

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        X = self.lr()
        Y = self.eval_pe()
        BS = self.batch_size()**4 * 100
        C = colorFromKinematic(self.kinematic())
        M = shapeFromLoss(self.alignment_loss())
        for x,y,s,c,m in zip(X,Y,BS,C,M):
            ax.scatter(x,y,s=s,c=c,marker=m)

        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Pointing Error [mrad]')
        ax.set_xscale('log')

        plt.show()
        plt.legend()

if __name__ == '__main__':
    csv_path = '/Users/Synhelion/Desktop/experiment_results.csv'
    csv_reader = ExperimentCSV()
    csv_reader.readCSV(csv_path=csv_path)
    csv_reader.plot()
