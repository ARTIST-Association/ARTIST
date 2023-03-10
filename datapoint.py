#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:32:12 2023

@author: user
"""
class DataPoint(object):
    def __init__(self,
                 point_id,
                 desired_image,
                 desired_concentrator_normal,
                 sun_directions):
        
        self.point_id = point_id
        self.desired_image = desired_image
        self.desired_concentrator_normal = desired_concentrator_normal
        self.sun_directions = sun_directions
        
    def __call__(self):
        return (self.desired_image, self.desired_concentrator_normal, self.sun_directions)
        
    def desired_image(self):
        return self.desired_image
    
    def desired_concentrator_normal(self):
        return self.desired_concentrator_normal
    
    def sun_directions(self):
        return self.sun_directions