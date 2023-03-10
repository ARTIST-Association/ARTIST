#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:30:04 2022

@author: moritz.leibauer@rwth-aachen.de (alt. moritz.leibauer@synhelion.com)
"""

import pandas as pd
import torch

class UCTrainingDataCSVTorch:
    _sep = ';' # csv data seperator

    # names
    _solar_pos_name_e = 'Solar Pos E [m]'
    _solar_pos_name_n = 'Solar Pos N [m]'
    _solar_pos_name_u = 'Solar Pos U [m]'
    _solar_diameter_name = 'Solar Diameter [m]'
    _target_normal_name_e = 'Target Normal E [m]'
    _target_normal_name_n = 'Target Normal N [m]'
    _target_normal_name_u = 'Target Normal U [m]'
    _target_side_name_e = 'Target Side E [m]'
    _target_side_name_n = 'Target Side N [m]'
    _target_side_name_u = 'Target Side U [m]'
    _target_up_name_e = 'Target Up E [m]'
    _target_up_name_n = 'Target Up N [m]'
    _target_up_name_u = 'Target Up U [m]'
    _target_center_name_e = 'Target Center E [m]'
    _target_center_name_n = 'Target Center N [m]'
    _target_center_name_u = 'Target Center U [m]'
    _target_width_name = 'Target Width [m]'
    _target_height_name = 'Target Height [m]'
    _axis_steps_name_1 = 'Axis 1 [Steps]'
    _axis_steps_name_2 = 'Axis 2 [Steps]'
    _img_path_name = 'Image'

    _gr1_name = 'GR1 []'
    _theta_k_name = 'Theta_k [mrad]'
    _gr2_name = 'GR2 []'
    _tau_k_name = 'Tau_k [mrad]'
    _alpha_name = 'Alpha [mrad]'
    _beta_name = 'Beta [mrad]'
    _gamma_name = 'Gamma [mrad]'
    _delta_name = 'Delta [mrad]'

    _heliostat_pos_name_e = 'Heliostat Position E [m]'
    _heliostat_pos_name_n = 'Heliostat Position N [m]'
    _heliostat_pos_name_u = 'Heliostat Position U [m]'

    _joint_offset_name_e = 'Joint Offset E [m]'
    _joint_offset_name_n = 'Joint Offset N [m]'
    _joint_offset_name_u = 'Joint Offset U [m]'

    _normal_offset_name_e = 'Normal Offset E [m]'
    _normal_offset_name_n = 'Normal Offset N [m]'
    _normal_offset_name_u = 'Normal Offset U [m]'
    
    def __init__(self, device):
        
        self._device = device

    def read(self, file_path: str):
        print('Reading training data from .csv file.')
        self._data = pd.read_csv(file_path, sep = self._sep)
        print('- got data of shape: ' + str(self._data.shape))
        print('Done')

    def solar_position(self):
        return [torch.tensor([e, n, u], device=self._device) 
                for e, n, u 
                in zip(self._data[self._solar_pos_name_e], self._data[self._solar_pos_name_n], self._data[self._solar_pos_name_u])]
    
    def solar_diameter(self):
        return torch.tensor(self._data[self._solar_diameter_name], device=self._device)

    def target_center(self):
        return [torch.Tensor([e, n, u], device=self._device) 
                for e, n, u 
                in zip(self._data[self._target_center_name_e], self._data[self._target_center_name_n], self._data[self._target_center_name_u])]

    def target_normal(self):
        return [torch.Tensor([e, n, u], device=self._device) 
                for e, n, u 
                in zip(self._data[self._target_normal_name_e], self._data[self._target_normal_name_n], self._data[self._target_normal_name_u])]

    def target_width(self):
        return [w for w in self._data[self._target_width_name]]
    
    def target_height(self):
        return [h for h in self._data[self._target_height_name]]
    
    def axis_steps(self):
        return [torch.Tensor([axis_1, axis_2]) for axis_1, axis_2 
                in zip(self._data[self._axis_steps_name_1], self._data[self._axis_steps_name_2])]
    
    def image_paths(self):
        return self._data[self._img_path_name]

    def save(self, filepath):
        self._data.to_csv(filepath, sep=self._sep)

if __name__ == '__main__':
    dsr = UCTrainingDataCSVTorch(torch.device('cpu'))
    print(dsr.solar_position())