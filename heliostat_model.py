#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:03:53 2023

@author: user
"""
import pytorch3d.transforms as throt
import torch
from typing import (
    Any,
    Dict,
    Iterable,
)

ParamGroups = Iterable[Dict[str, Any]]

class HeliostatModel:
    def __init__(self, alignment_model, concentrator_model):
        self.alignment_model = alignment_model
        self.concentrator_model = concentrator_model
        
        
        self.device = alignment_model._device
        # self.discrete_points = alignment_model.discrete_points
        # self.normals = alignment_model.normals
        # self._normals_ideal = alignment_model._normals_ideal
        
    def get_to_optimize(self):
        to_optimize = list(self.alignment_model.getTrainingParams().keys())
        #to_optimize += self.concentrator_model.getTrainingParams()
        return to_optimize
    
    def align(self, datapoint):
        sun_direction = datapoint.sun_directions()
        alignment_dict = self.alignment_model.alignFromDataPoint(datapoint)
        alignment = torch.stack([alignment_dict['side_up'], alignment_dict['side_east'], alignment_dict['normal']])
        pivot_point = alignment_dict['pivoting_point']
        return alignment, pivot_point, alignment_dict["normal_target"]
    
    def surface_points(self, alignment, pivot_point):
        align_origin = [
            throt.Rotate(alignment, dtype=alignment.dtype)]
        
        H_aligned = self.concentrator_model.align(alignment, align_origin, pivot_point)
        surface_points = H_aligned.discrete_points
        surface_normals = H_aligned.normals
        return surface_points, surface_normals
    
    # check
    def step(self, verbose):
        return self.concentrator_model.step(verbose)
    
    def get_params(self) -> ParamGroups:
        opt_params = []
        for key, value in self.alignment_model.getTrainingParams().items():
            opt_params.append({'params': value, 'name': key})
        return opt_params

    
    
    
    