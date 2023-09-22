#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:03:53 2023

@author: user
"""

class HeliostatModel:
    def __init__(self, alignment_model, concentrator_model):
        self.alignment_model = alignment_model
        self.concentrator_model = concentrator_model
        
        
        self.device = alignment_model.device
        # self.discrete_points = alignment_model.discrete_points
        # self.normals = alignment_model.normals
        # self._normals_ideal = alignment_model._normals_ideal
        
    def get_params(self):
        params = self.alignment_model.get_params()
        #params += self.concentrator_model.get_params()
        return params
    def get_to_optimize(self):
        to_optimize = self.alignment_model.get_to_optimize()
        #to_optimize += self.concentrator_model.get_to_optimize()
        return to_optimize
    def align(self, datapoint):
        return self.alignment_model.align2(datapoint.sun_directions)
    def surface_points(self, alignment, align_origin):
        H_aligned = self.concentrator_model.align(alignment, align_origin)
        surface_points = H_aligned.discrete_points
        surface_normals = H_aligned.normals
        return surface_points, surface_normals
    
    # check
    def step(self, verbose):
        return self.alignment_model.step(verbose)

    
    
    
    