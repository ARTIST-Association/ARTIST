# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 12:29:37 2022

@author: parg_ma
"""
import os
import copy
from defaults import get_cfg_defaults, load_config_file
def load_config(config_file_name):
    cfg_default = get_cfg_defaults()
    if config_file_name:
        print(f"load: {config_file_name}")
        # config_file = os.path.join("configs", config_file_name)
        cfg = load_config_file(cfg_default, config_file_name)
    else:
        print("No config loaded. Use defaults")
        cfg = cfg_default
    cfg.freeze()
    return cfg

initial_filename = "Nature_Distance_Sweep.yaml"
initial_folder = "TestingConfigs"
result_folder = "Results"
sweep_name = "Distance_Nature_Sweep"

num_images_array = [2,4,8]
distance_array = [400]

initial_path =  os.path.join(initial_folder, initial_filename)
result_path = os.path.join(result_folder,sweep_name)
os.makedirs(result_path, exist_ok=True)

cfg = load_config(initial_path)
# print(cfg)


            
            
for distance in distance_array:
    #initialize cfg and namespace
    new_cfg = copy.deepcopy(cfg)
    name_string =f"{sweep_name}_"
    #start loop
    short_name = "D"
    long_name = "H.DEFLECT_DATA.POSITION_ON_FIELD"
    cor_distance = distance
    if distance == 400:
        new_cfg.merge_from_list(["AC.RECEIVER.PLANE_X",new_cfg.AC.RECEIVER.PLANE_X*3])
        new_cfg.merge_from_list(["AC.RECEIVER.PLANE_Y",new_cfg.AC.RECEIVER.PLANE_Y*3])
    new_cfg.merge_from_list([long_name, [13.2, distance, 1.795]])
    name_string = name_string+short_name+"_"+str(distance)+"_" 
    for num_images in num_images_array:
        short_name = "I"
        long_name = "TRAIN.SUN_DIRECTIONS.RAND.NUM_SAMPLES"
        new_cfg.merge_from_list([long_name, num_images])
        name_string_final = name_string+short_name+"_"+str(num_images)+"_" 
        #Save to File
        sweep_path = os.path.join(result_path, name_string_final)
        print(sweep_path)
        os.makedirs(sweep_path, exist_ok=True)
        with open(os.path.join(sweep_path, "config.yaml"), "w") as f:
            f.write(new_cfg.dump())  # cfg, f, default_flow_style=False)