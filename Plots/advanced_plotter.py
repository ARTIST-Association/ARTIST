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
import functools
from build_heliostat_model import build_target_heliostat, build_heliostat, load_heliostat
import hausdorff_distance
import data
from yacs.config import CfgNode
from defaults import get_cfg_defaults, load_config_file


from data import generate_sun_array
from environment import Environment
import hausdorff_distance
from heliostat_models import Heliostat
from render import Renderer
from build_heliostat_model import build_target_heliostat, build_heliostat, load_heliostat

import disk_cache
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

join_paths = cast(
    Callable[[List[str]], str],
    functools.partial(functools.reduce, os.path.join),
)

import training
from heliostat_models import Heliostat
from yacs.config import CfgNode

def colorbar(mappable: cm.ScalarMappable) -> matplotlib.colorbar.Colorbar:
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

@torch.no_grad()
class Plotter():
    @torch.no_grad()
    def __init__(self, cfg, R,sun_directions, loss_func, logdir, device, train_prealignment= None, test_prealignment=None):
        self.plot_real_data = True
                
        (
            (
                cached_generate_grid_sun_array,
                cached_generate_spheric_sun_array,
                cached_generate_season_sun_array,
            ),
            (
                cached_generate_grid_dataset,
                cached_generate_naive_grid_dataset,
                cached_generate_spheric_dataset,
                cached_generate_naive_spheric_dataset,
                cached_generate_season_dataset,
                cached_generate_naive_season_dataset
                
            ),
        ) = dataset_cache.set_up_test_dataset_caching(device, None)
        
        cached_build_target_heliostat = cast(
            Callable[[CfgNode, torch.Tensor, th.device], Heliostat],
            disk_cache.disk_cache(
                build_target_heliostat,
                device,
                'cached',
                ignore_argnums=[2],
            ),
        )
        
        H_validation = cached_build_target_heliostat(
        cfg, sun_directions, device)
        ENV_validation = Environment(cfg.AC, device)
        
        
        if self.plot_real_data: 
            parent_folder_path_train = ["..\\"+path for path in cfg.TRAIN.IMAGES.PATHS]
            assert cfg.TEST.SUN_DIRECTIONS.CASE == "vecs", \
                'to plot real data, sun directions must be given by CASE "vecs".'
            print("Initialize Real Data Plot")
 
            train_sundirections = th.tensor(cfg.TRAIN.SUN_DIRECTIONS.VECS.DIRECTIONS)
            H_naive_trainset = build_target_heliostat(cfg, train_sundirections , device)
            H_naive_trainset._normals = H_naive_trainset.get_raw_normals_ideal()
            train_targets = data.load_images(
                parent_folder_path_train,
                cfg.AC.RECEIVER.PLANE_X,
                cfg.AC.RECEIVER.PLANE_Y,
                cfg.AC.RECEIVER.RESOLUTION_X,
                cfg.AC.RECEIVER.RESOLUTION_Y,
                device,
                'train',
                None,
            )
            train_target_sets = hausdorff_distance.images_to_sets(
                train_targets,
                cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VALS,
                cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VAL_RADIUS,
            )
            self.trainset_objects = training.TestObjects(
                None,
                ENV_validation,
                R,
                train_targets,
                train_target_sets,
                train_sundirections,
                loss_func,
                cfg,
                None,#epoch
                "test_trainset",
                None, #writer
                H_naive_trainset,
                None,
                False,#Reduction
                train_prealignment if train_prealignment is not None else None
                )
            
            naive_train_objects = self.trainset_objects._replace(H=H_naive_trainset)
            self.naive_train_loss, self.naive_hd_train, self.naive_train_targets = training.test_batch(naive_train_objects)
            # plt.imshow(self.naive_train_targets[2])
            # plt.show()
            # exit()
            parent_folder_path_test = ["..\\"+path for path in cfg.TEST.IMAGES.PATHS]
            test_sundirections = th.tensor(cfg.TEST.SUN_DIRECTIONS.VECS.DIRECTIONS)
            H_naive_testset = build_target_heliostat(cfg, test_sundirections , device)
            H_naive_testset._normals = H_naive_testset.get_raw_normals_ideal()
            test_targets = data.load_images(
                parent_folder_path_test,
                cfg.AC.RECEIVER.PLANE_X,
                cfg.AC.RECEIVER.PLANE_Y,
                cfg.AC.RECEIVER.RESOLUTION_X,
                cfg.AC.RECEIVER.RESOLUTION_Y,
                device,
                'test',
                None,
            )
            test_target_sets = hausdorff_distance.images_to_sets(
                test_targets,
                cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VALS,
                cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VAL_RADIUS,
            )
            self.testset_objects = training.TestObjects(
                None,
                ENV_validation,
                R,
                test_targets,
                test_target_sets,
                test_sundirections,
                loss_func,
                cfg,
                None,#epoch
                "test_testset",
                None, #writer
                H_naive_testset,
                None,
                False,#Reduction
                test_prealignment if test_prealignment is not None else None
                )
            
            naive_test_objects = self.testset_objects._replace(H=H_naive_testset)
            self.naive_test_loss, self.naive_hd_test, self.naive_test_targets = training.test_batch(naive_test_objects)
            
    def season_plot(
            self,
            H,
            logdir
    ) -> None:
        epoch = 0
        so = self.season_test_objects._replace(H=H, epoch=epoch)
        season_test_loss, season_test_hd, season_test_bitmaps = training.test_batch(
            so
        )
        transform = torchvision.transforms.CenterCrop(140)
        
        ground_truth = transform(so.test_targets)
        

        ideal = transform(self.season_naive_test_targets)
        ideal_loss = self.season_naive_test_loss
        
        prediction = transform(season_test_bitmaps)
        prediction_loss = season_test_loss
        
        colormap = "afmhot"
        hg_l = 1.5 #high lighting 
        fig = plt.figure(figsize=(7.6, 8))
        gs = GridSpec(3, 3)  # width_ratios=[1, 1, 1], height_ratios=[1.0, 0.05])
        plt.subplots_adjust(wspace=0.0001)
        for (i, img) in enumerate(ground_truth):
            ax = fig.add_subplot(gs[i, 0])
            v_max = th.max(th.stack((ground_truth[i]**hg_l,ideal[i]**hg_l, prediction[i]**hg_l)))
            print(th.max(ground_truth[i]**hg_l),th.max(ideal[i]**hg_l), th.max(prediction[i]**hg_l))
            ax.imshow(ground_truth[i]**hg_l, cmap=colormap, vmax=v_max)
            ax.axis('off')
            if i == 0:
                ax.set_title("Ground Truth", size=18)
                string = 'Shortest Day'+'\n'+' 9:00 a.m.'
                ax.text(
                    -0.15,
                    0.5,
                    string,
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=90,
                    size=18,
                    transform=ax.transAxes,
                )
            if i == 1:
                string = 'Equinox'+'\n'+' 12:00 a.m.'
                ax.text(
                    -0.15,
                    0.5,
                    string,
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=90,
                    size=18,
                    transform=ax.transAxes,
                )
            if i == 2:
                string = 'Longest Day'+'\n'+' 18:00 a.m.'
                ax.text(
                    -0.15,
                    0.5,
                    string,
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=90,
                    size=18,
                    transform=ax.transAxes,
                )
        #ideal
            ax = fig.add_subplot(gs[i, 1])
            ax.imshow(ideal[i]**hg_l, cmap=colormap, vmax=v_max)
            ax.axis('off')
            ax.text(
                0.5,
                -0.10,
                f"L1: {ideal_loss[i].item():.2e}",
                size=18,
                ha="center",
                transform=ax.transAxes,
            )
            if i == 0:
                ax.set_title("Ideal", size=18)
        #prediction
            ax = fig.add_subplot(gs[i, 2])
            ax.imshow(prediction[i]**hg_l, cmap=colormap, vmax=v_max)
            ax.axis('off')
            ax.text(
                0.5,
                -0.10,
                f"L1: {prediction_loss[i].item():.2e}",
                size=18,
                ha="center",
                transform=ax.transAxes,
                fontweight="bold",
            )
            if i == 0:
                ax.set_title("Prediction", size=18)
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(logdir, f"season_test"), dpi=fig.dpi)
        plt.close(fig)
    
    def real_data_plot(
            self,
            H,
            logdir,
            H_target = None
    ) -> None:
        epoch = 0
        O_train = self.trainset_objects._replace(H=H, epoch=epoch)
        train_loss, train_hd, train_bitmaps = training.test_batch(
            O_train
            )
        O_test  = self.testset_objects._replace(H=H, epoch=epoch)
        test_loss, test_hd, test_bitmaps = training.test_batch(
            O_test
            )
        
        
        H._normals = H_target._normals
        O_train_deflec = self.trainset_objects._replace(H=H, epoch=epoch)
        train_loss_deflec, train_hd_deflec, train_bitmaps_deflec = training.test_batch(
            O_train
            )
        O_test_deflec = self.testset_objects._replace(H=H, epoch=epoch)
        test_loss_deflec, test_hd_deflec, test_bitmaps_deflec = training.test_batch(
            O_test
            )
            

        
        ground_truth_train = O_train.test_targets
        ground_truth_test  = O_test.test_targets

        ideal_train = self.naive_train_targets
        ideal_train_loss = self.naive_train_loss
        
        ideal_test = self.naive_test_targets
        ideal_test_loss = self.naive_test_loss
        
        deflec_train = train_bitmaps_deflec
        deflec_train_loss = train_loss_deflec
        
        deflec_test = test_bitmaps_deflec
        deflec_test_loss = test_loss_deflec
        
        prediction_train = train_bitmaps
        prediction_train_loss = train_loss
        
        prediction_test = test_bitmaps
        prediction_test_loss = test_loss
        
        colormap = "afmhot"
        hg_l = 1.5 #high lighting 
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(4, 4)  # TODO len(train+test)
        plt.subplots_adjust(wspace=0.0001)
        for (i, img) in enumerate(ground_truth_train):
            ax = fig.add_subplot(gs[i, 0])
            v_max = th.max(th.stack((ground_truth_train[i]**hg_l,ideal_train[i]**hg_l, deflec_train[i]**hg_l,prediction_train[i]**hg_l)))
            ax.imshow(ground_truth_train[i]**hg_l, cmap=colormap, vmax = v_max)
            ax.axis('off')
            if i == 0:
                ax.set_title("Ground Truth", size=18)
                ax.text(
                    -0.15,
                    1.1,
                    "Train",
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontweight="bold",
                    size=18,
                    transform=ax.transAxes,
                )
                string = '24.03.22'+'\n'+' 10:09 a.m.'
                ax.text(
                    -0.15,
                    0.5,
                    string,
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=90,
                    size=18,
                    transform=ax.transAxes,
                )
            if i == 1:
                string = '24.03.22'+'\n'+' 04:09 p.m.'
                ax.text(
                    -0.15,
                    0.5,
                    string,
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=90,
                    size=18,
                    transform=ax.transAxes,
                )
        #ideal
            ax = fig.add_subplot(gs[i, 1])
            ax.imshow(ideal_train[i]**hg_l, cmap=colormap, vmax = v_max)
            ax.axis('off')
            ax.text(
                0.5,
                -0.10,
                f"L1: {ideal_train_loss[i].item():.2e}",
                size=18,
                ha="center",
                transform=ax.transAxes,
            )
            if i == 0:
                ax.set_title("Ideal", size=18)
        #deflectometry
            ax = fig.add_subplot(gs[i, 2])
            ax.imshow(deflec_train[i]**hg_l, cmap=colormap, vmax = v_max)
            ax.axis('off')
            ax.text(
                0.5,
                -0.10,
                f"L1: {deflec_train_loss[i].item():.2e}",
                size=18,
                ha="center",
                transform=ax.transAxes,
            )
            if i == 0:
                ax.set_title("Deflectometry", size=18)        
        #prediction
            ax = fig.add_subplot(gs[i, 3])
            ax.imshow(prediction_train[i]**hg_l, cmap=colormap, vmax = v_max)
            ax.axis('off')
            ax.text(
                0.5,
                -0.10,
                f"L1: {prediction_train_loss[i].item():.2e}",
                size=18,
                ha="center",
                transform=ax.transAxes,
                fontweight="bold",
            )
            if i == 0:
                ax.set_title("Prediction", size=18)
                
        for (i, img) in enumerate(ground_truth_test):
            tl = len(ground_truth_train) #trainset length
            ax = fig.add_subplot(gs[i+tl, 0])
            v_max = th.max(th.stack((ground_truth_test[i]**hg_l,ideal_test[i]**hg_l,deflec_test[i]**hg_l, prediction_test[i]**hg_l)))
            ax.imshow(ground_truth_test[i]**hg_l, cmap=colormap, vmax = v_max)
            ax.axis('off')
            if i == 0:
                string = '08.11.22'+'\n'+' 03:16 p.m.'
                ax.text(
                    -0.15,
                    0.5,
                    string,
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=90,
                    size=18,
                    transform=ax.transAxes,
                )
                ax.text(
                    -0.15,
                    1.1,
                    "Test",
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontweight="bold",
                    size=18,
                    transform=ax.transAxes,
                )
            if i == 1:
                string = '08.11.22'+'\n'+' 02:09 p.m.'
                ax.text(
                    -0.15,
                    0.5,
                    string,
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=90,
                    size=18,
                    transform=ax.transAxes,
                )
        #ideal
            ax = fig.add_subplot(gs[i+tl, 1])
            ax.imshow(ideal_test[i]**hg_l, cmap=colormap, vmax = v_max)
            ax.axis('off')
            ax.text(
                0.5,
                -0.10,
                f"L1: {ideal_test_loss[i].item():.2e}",
                size=18,
                ha="center",
                transform=ax.transAxes,
            )
        #deflectometry
            ax = fig.add_subplot(gs[i+tl, 2])
            ax.imshow(deflec_test[i]**hg_l, cmap=colormap, vmax = v_max)
            ax.axis('off')
            ax.text(
                0.5,
                -0.10,
                f"L1: {deflec_test_loss[i].item():.2e}",
                size=18,
                ha="center",
                transform=ax.transAxes,
            )
        #prediction
            ax = fig.add_subplot(gs[i+tl, 3])
            ax.imshow(prediction_test[i]**hg_l, cmap=colormap, vmax = v_max)
            ax.axis('off')
            ax.text(
                0.5,
                -0.10,
                f"L1: {prediction_test_loss[i].item():.2e}",
                size=18,
                ha="center",
                transform=ax.transAxes,
                fontweight="bold",
            )
        
        
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(logdir, f"real_data"), dpi=fig.dpi)
        plt.close(fig)
    
    
    def create_plots(self, H, logdir, H_target=None):
            self.real_data_plot(H, logdir, H_target)

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

def main(path, use_prealignment):
    path_to_config = path.copy()
    path_to_model = path.copy()
    path_to_model.extend(['Logfiles', 'MultiNURBSHeliostat.pt'])
    path_to_config.extend(['config.yaml'])
    MODEL_PATH = os.path.join(*path_to_model)
    CONFIG_PATH = os.path.join(*path_to_config)
    LOGDIR = ""
    
    cfg_default = get_cfg_defaults()
    print(f"load: {CONFIG_PATH}")
    cfg = load_config_file(cfg_default, CONFIG_PATH)
    cfg.merge_from_list(['CP_PATH', MODEL_PATH])
    cfg.merge_from_list(['USE_NURBS', False])
    cfg.freeze()
    
    th.manual_seed(cfg.SEED)
    device = th.device(
        'cuda'
        if cfg.USE_GPU and th.cuda.is_available()
        else 'cpu'
        )
    if use_prealignment ==  True:
        prealignment = [ #converged prealigment from pretraining. its hardcoded here since it is not saved in training. Remove later
            [th.tensor(-0.0140), th.tensor(0.0085), th.tensor(-0.7635)],
            [th.tensor(0.0127), th.tensor(0.0108), th.tensor(0.0631)], 
                    ]
        test_prealignment = [
            [th.tensor(0.0038), th.tensor(0.0010), th.tensor(-0.7040)], 
            [th.tensor(-0.0243), th.tensor(-0.0014), th.tensor(-0.6096)], 
            ]
    else:
        prealignment = None
        test_prealignment = None
    
    
    loss_func = get_loss_func(cfg.TRAIN.LOSS)
    sun_directions, ae = generate_sun_array(
        cfg.TRAIN.SUN_DIRECTIONS, device)
    H_trained  = load_heliostat(cfg, sun_directions, device)
    H_target   = build_target_heliostat(cfg, sun_directions, device)
    
    ENV = Environment(cfg.AC, device)
    R = Renderer(H_trained, ENV)
    
    plot = Plotter(cfg, R, sun_directions, loss_func, LOGDIR, device, train_prealignment= prealignment, test_prealignment=test_prealignment)


    plot.create_plots(H_trained, LOGDIR, H_target = H_target)




if __name__ == '__main__':
    path = ['..','Results', 'ForRealData', '3I7N']
    resolution = 256
    use_prealignment = True
    main(path, use_prealignment)