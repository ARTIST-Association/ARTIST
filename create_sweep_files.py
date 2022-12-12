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

initial_filename = "ForNextPaper.yaml"
initial_folder = "TestingConfigs"
result_folder = "NextPaperSweep"
sweep_name = "test"

wd_array = [0.1,0.01,0.001]
lr_array = [1e-4,1e-5]
sd_array = [2,3,4]
n_array  = [5,6,8,10,12]

initial_path =  os.path.join(initial_folder, initial_filename)
result_path = os.path.join(initial_folder, result_folder)
os.makedirs(result_path, exist_ok=True)

cfg = load_config(initial_path)
print(cfg)

for wd in wd_array:
    #initialize cfg and namespace
    new_cfg = copy.deepcopy(cfg)
    name_string =f"{sweep_name}_"
    #start loop
    
    short_name = "wd"
    long_name = "TRAIN.OPTIMIZER.WEIGHT_DECAY"
    new_cfg.merge_from_list([long_name, wd])
    name_string = name_string+short_name+"_"+str(wd)+"_" 
    
    for sd in sd_array:
        short_name = "sd"
        long_name = "NURBS.SPLINE_DEGREE"
        
        new_cfg.merge_from_list([long_name, sd])
        name_string_sd = name_string+short_name+"_"+str(sd)+"_" 
        for n in n_array:
            short_name = "N"
            long_name_1 = "NURBS.ROWS"
            long_name_2 = "NURBS.COLS"
            
            new_cfg.merge_from_list([long_name_1, n])
            new_cfg.merge_from_list([long_name_2, n])
            name_string_n = name_string_sd+short_name+"_"+str(n)+"_" 
            for lr in lr_array:
                short_name = "lr"
                long_name = "TRAIN.OPTIMIZER.LR"       
                
                new_cfg.merge_from_list([long_name, lr])
                name_string_final = name_string_n+short_name+"_"+str(lr) #no underline in last for loop
                
                with open(os.path.join(result_path, name_string_final+".yaml"), "w") as f:
                    f.write(new_cfg.dump())  # cfg, f, default_flow_style=False)
        

wd = []