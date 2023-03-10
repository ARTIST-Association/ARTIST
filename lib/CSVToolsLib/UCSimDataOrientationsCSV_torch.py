#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:30:04 2022

@author: moritz.leibauer@rwth-aachen.de (alt. moritz.leibauer@synhelion.com)
"""

import pandas as pd
import torch

class UCSimDataOrientationsCSVTorch:
    _solar_pos_name_e = 'Solar Pos E [m]'
    _solar_pos_name_n = 'Solar Pos N [m]'
    _solar_pos_name_u = 'Solar Pos U [m]'

    _target_normal_name_e = 'Target Normal E [m]'
    _target_normal_name_n = 'Target Normal N [m]'
    _target_normal_name_u = 'Target Normal U [m]'

    _target_center_name_e = 'Target Center E [m]'
    _target_center_name_n = 'Target Center N [m]'
    _target_center_name_u = 'Target Center U [m]'

    def __init__(self, file_path):
        self._data = pd.read_csv(file_path, sep = self._sep)

    def solar_position(self):
        return [torch.tensor([e, n, u], device=self._device) 
                for e, n, u 
                in zip(self._data[self._solar_pos_name_e], self._data[self._solar_pos_name_n], self._data[self._solar_pos_name_u])]

    def target_center(self):
        return [torch.Tensor([e, n, u], device=self._device) 
                for e, n, u 
                in zip(self._data[self._target_center_name_e], self._data[self._target_center_name_n], self._data[self._target_center_name_u])]

    def target_normal(self):
        return [torch.Tensor([e, n, u], device=self._device) 
                for e, n, u 
                in zip(self._data[self._target_normal_name_e], self._data[self._target_normal_name_n], self._data[self._target_normal_name_u])]