import sys
sys.path.append('../')
import datetime
import os
import time as to_time
from typing import Any, Dict, List, Optional, Union
import dataset_cache
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import torch as th
import torchvision
from torch.utils.tensorboard import SummaryWriter
from environment import Environment
from heliostat_models import AbstractHeliostat
import utils

from build_heliostat_model import build_target_heliostat, build_heliostat, load_heliostat
import hausdorff_distance
import data
from defaults import get_cfg_defaults, load_config_file
import disk_cache
from data import generate_sun_array
from render import Renderer
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
import training
from heliostat_models import Heliostat
from yacs.config import CfgNode

def get_loss_func(
        cfg_loss: CfgNode,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    name = cfg_loss.NAME.lower()
    if name == "mse":
        loss_func = th.nn.MSELoss()
    elif name == "l1":
        loss_func = th.nn.L1Loss()
    else:
        raise ValueError(
            "Loss function name not found, change name or implement new loss")
    return loss_func

def get_normals(heliostat_target, heliostat_pred):
        target_normal_vecs = heliostat_target.normals#.float()
        ideal_normal_vecs = heliostat_target._normals_ideal.double()
        pred_normal_vecs = heliostat_pred.normals
    
        target_angles = th.acos(
            th.clip(th.sum(ideal_normal_vecs * target_normal_vecs, dim=-1), -1, 1),
        ).detach().cpu()
        pred_angles = th.acos(
            th.clip(th.sum(ideal_normal_vecs * pred_normal_vecs, dim=-1), -1, 1),
        ).detach().cpu()
    
        target_angles = target_angles - th.min(target_angles)
        pred_angles = pred_angles - th.min(pred_angles)
    
        diff_angles = abs(target_angles - pred_angles)
    
        # Get discrete points
        target_points = heliostat_target.discrete_points
        target_points = target_points.detach().cpu()
    
        pred_points = heliostat_pred.discrete_points
        pred_points = pred_points.detach().cpu()
        diff_points = pred_points.clone()
    
        target_points[:, 2] = target_angles / 1e-3
        pred_points[:, 2] = pred_angles / 1e-3
        diff_points[:, 2] = diff_angles / 1e-3
        # print(th.max(pred_angles), th.mean(pred_angles))
        target = target_points
        pred = pred_points
        # pred = pred[pred[:, -1] >= th.max(target[:, -1])]
        diff = diff_points
        return target, pred, diff

@th.no_grad()
def plot_surfaces_mrad(
       H_image,
       H_distance,
) -> None:
    fig = plt.figure(figsize=(6, 10))
    gs = GridSpec(5, 4, height_ratios=[1,1,1,1,0.1])
    colormap="magma"
    for j in range(2):
        if j==0: # left column    
            title = ["","2","4","8","16","32"]
            H = H_image
            position_x = -0.1
            position_y = 0.5
            alignment = "right"
            sub_title = "# Images\n at real position\n in heliostat field"
            sub_alignment = 0.5
            start = 0
            end = 2
        if j==1: # right column
            title = ["","50m","100m","200m","400m"]
            H = H_distance
            position_x = 1.1
            position_y = 0.5
            sub_title = "Distance\nto tower\nat 16 images"
            sub_alignment = 0.5
            alignment = "left"
            start = 2
            end = 4
        trained, target = H
        for (i, (heliostat_target,heliostat_pred)) in enumerate(zip(target,trained)):
            target, pred, diff = get_normals(heliostat_target, heliostat_pred)
            if i == 0 and j==0: #plot Groundtruth once
                ax = fig.add_subplot(gs[i, 1:3])
                im1 = ax.scatter(target[:, 0], target[:, 1], c=target[:, 2], cmap=colormap)
                ax.set_xlim(th.min(target[:, 0]), th.max(target[:, 0]))
                ax.set_ylim(th.min(target[:, 1]), th.max(target[:, 1]))
                ax.set_title('Measured surface [mrad]', size=18)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                vmax = max(target[:,2])
                
            ax = fig.add_subplot(gs[i+1, start:end])
            if i ==0:
                ax.text(sub_alignment, 1.3, sub_title, horizontalalignment=alignment,
                     verticalalignment='center', size=18, transform=ax.transAxes)    
            im1 = ax.scatter(pred[:, 0], pred[:, 1], c=pred[:, 2], cmap=colormap, vmax=vmax)
            ax.set_xlim(th.min(pred[:, 0]), th.max(pred[:, 0]))
            ax.set_ylim(th.min(pred[:, 1]), th.max(pred[:, 1]))
            ax.text(position_x, position_y, title[i+1], horizontalalignment=alignment,
                 verticalalignment='center', size=18, transform=ax.transAxes)    
            # ax1.title.set_text('Original Surface [mrad]')
            # ax1.set_aspect("equal")
            # if i==1:
            #     ax.text(1.05, 1.2, "Distance to \n Tower [m]", horizontalalignment='left',
            #          verticalalignment='center', size=18, transform=ax.transAxes)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            # if iSie

    cax = fig.add_subplot(gs[-1, 0:])
    cax.tick_params(labelsize=18)
    cb = plt.colorbar(im1, cax=cax, orientation="horizontal")
    cb.set_label(label='mrad', size=18)
    plt.savefig("surface_distance_nature.png", dpi=fig.dpi, bbox_inches = "tight")
    plt.show()
    

def create_heliostats(paths):
    H_trained_array = []
    H_target_array = []
    for path in paths:
        print(path)

        path_to_config = path.copy()
        path_to_model = path.copy()
        path_to_model.extend(['Logfiles', 'MultiNURBSHeliostat.pt'])
        path_to_config.extend(['config.yaml'])
        MODEL_PATH = os.path.join(*path_to_model)
        print(MODEL_PATH)
        CONFIG_PATH = os.path.join(*path_to_config)
        exit()       
        cfg_default = get_cfg_defaults()
        print(f"load: {CONFIG_PATH}")
        cfg = load_config_file(cfg_default, CONFIG_PATH)
        cfg.merge_from_list(['CP_PATH', MODEL_PATH])
        cfg.merge_from_list(['H.DEFLECT_DATA.DIRECTORY', os.path.join("..", cfg.H.DEFLECT_DATA.DIRECTORY)])
        # cfg.merge_from_list(['USE_NURBS', False])
        # cfg.merge_from_list(['H.DEFLECT_DATA.TAKE_N_VECTORS', 800])
        cfg.freeze()
        
        th.manual_seed(cfg.SEED)
        device = th.device(
            'cuda'
            if cfg.USE_GPU and th.cuda.is_available()
            else 'cpu'
            )
        
        
        sun_directions, ae = generate_sun_array(
            cfg.TRAIN.SUN_DIRECTIONS, device)
        H_trained  = load_heliostat(cfg, sun_directions, device)
        H_target   = build_target_heliostat(cfg, sun_directions, device)

        H_trained_array.append(H_trained)
        H_target_array.append(H_target)
    return H_trained_array, H_target_array

def main(paths_image, paths_distance):
    H_images =create_heliostats(paths_image)
    H_distances = create_heliostats(paths_distance)
    plot_surfaces_mrad(
            H_images,
            H_distances,
            )
if __name__ == '__main__':
    paths_num_im = [
        ['..','Results', 'DistanceWithFocus', '_DistanceWithFocusN7S3_D_25_I_2_' ],
        ['..','Results', 'DistanceWithFocus', '_DistanceWithFocusN7S3_D_25_I_4_' ],
        ['..','Results', 'DistanceWithFocus', '_DistanceWithFocusN7S3_D_25_I_8_' ],
        ]
    paths_distance = [
        ['..','Results', 'DistanceWithFocus', '_DistanceWithFocusN7S3_D_50_I_16_' ],
        ['..','Results', 'DistanceWithFocus', '_DistanceWithFocusN7S3_D_100_I_16_' ],
        ['..','Results', 'DistanceWithFocus', '_DistanceWithFocusN7S3_D_200_I_16_' ],
        ]

    main(paths_num_im, paths_distance)
    
