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
from render import Renderer

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
class Plotter:
    @torch.no_grad()
    def __init__(
            self,
            cfg: CfgNode,
            R: Renderer,
            sun_directions: torch.Tensor,
            loss_func: Callable[
                [torch.Tensor, torch.Tensor],
                torch.Tensor,
            ],
            device: th.device,
            train_prealignment: Optional[List[List[torch.Tensor]]] = None,
            test_prealignment: Optional[List[List[torch.Tensor]]] = None,
    ) -> None:
        cfg_test = cfg.TEST
        self.plot_grid = cfg_test.PLOT.GRID
        self.plot_season = cfg_test.PLOT.SEASON
        self.plot_spheric = cfg_test.PLOT.SPHERIC
        self.plot_real_data = cfg_test.PLOT.REAL_DATA
                
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
        
        if cfg.TEST.PLOT.GRID:
            print("Create Dataset for Grid Plot")
            (
                self.grid_test_sun_directions,
                self.grid_test_ae,
            ) = cached_generate_grid_sun_array(
                cfg.TEST.SUN_DIRECTIONS,
                device,
                case="grid",
            )
            self.grid_test_targets = cached_generate_grid_dataset(
                H_validation,
                ENV_validation,
                self.grid_test_sun_directions,
                None,
                "grid",
            )
            # # th.random.set_rng_state(state)
            H_naive_grid = cached_build_target_heliostat(
                cfg, sun_directions, device)
            H_naive_grid._normals = H_naive_grid.get_raw_normals_ideal()
            self.grid_naive_targets = cached_generate_naive_grid_dataset(
                H_naive_grid,
                ENV_validation,
                self.grid_test_sun_directions,
                None,
                "naive",
            )
        if cfg.TEST.PLOT.SPHERIC:
            print("Create Dataset for Spheric Plot")
            (
                self.spheric_test_sun_directions,
                self.spheric_test_ae,
            ) = cached_generate_spheric_sun_array(
                cfg.TEST.SUN_DIRECTIONS,
                device,
                train_vec=sun_directions,
                case="spheric",
            )
            self.spheric_test_targets = cached_generate_spheric_dataset(
                H_validation,
                ENV_validation,
                self.spheric_test_sun_directions,
                None,
                "spheric",
            )

            H_naive_spheric = cached_build_target_heliostat(
                cfg, sun_directions, device)
            H_naive_spheric._normals = H_naive_spheric.get_raw_normals_ideal()
            self.spheric_naive_test_targets = cached_generate_naive_spheric_dataset(
                H_naive_spheric,
                ENV_validation,
                self.spheric_test_sun_directions,
                None,
                "naive_spheric",
            )
        if cfg.TEST.PLOT.SEASON:
            print("Create Dataset for Season Plot")
            (
                season_test_sun_directions,
                season_test_extras,
            ) = cached_generate_season_sun_array(
                cfg.TEST.SUN_DIRECTIONS,
                device,
                case="season",
            )
            # TODO bring to GPU in data.py
            season_test_sun_directions = season_test_sun_directions.to(device)
            season_test_targets = cached_generate_season_dataset(
                H_validation,
                ENV_validation,
                season_test_sun_directions,
                None,
                "season",
            )
            H_naive_season = cached_build_target_heliostat(
                cfg, sun_directions, device)
            H_naive_season._normals = H_naive_season.get_raw_normals_ideal()
            
            season_test_target_sets = hausdorff_distance.images_to_sets(
                season_test_targets,
                cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VALS,
                cfg.TRAIN.LOSS.HAUSDORFF.CONTOUR_VAL_RADIUS,
            )
            
            self.season_test_objects = training.TestObjects(
                None,
                ENV_validation,
                R,
                season_test_targets,
                season_test_target_sets,
                season_test_sun_directions,
                loss_func,
                cfg,
                None,  # epoch
                "season_test",
                None,  # writer
                H_validation,
                None,
                False,  # Reduction
                None,
            )
            
            season_test_objects = self.season_test_objects._replace(H=H_naive_season)
            self.season_naive_test_loss, self.season_naive_hd, self.season_naive_test_targets = training.test_batch(season_test_objects)
        if cfg.TEST.PLOT.REAL_DATA:
            assert cfg.TEST.SUN_DIRECTIONS.CASE == "vecs", \
                'to plot real data, sun directions must be given by CASE "vecs".'
            print("Initialize Real Data Plot")
 
            train_sundirections = th.tensor(cfg.TRAIN.SUN_DIRECTIONS.VECS.DIRECTIONS)
            H_naive_trainset = build_target_heliostat(cfg, train_sundirections , device)
            H_naive_trainset._normals = H_naive_trainset.get_raw_normals_ideal()
            train_targets = data.load_images(
                cfg.TRAIN.IMAGES.PATHS,
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
            
            test_sundirections = th.tensor(cfg.TEST.SUN_DIRECTIONS.VECS.DIRECTIONS)
            H_naive_testset = build_target_heliostat(cfg, test_sundirections , device)
            H_naive_testset._normals = H_naive_testset.get_raw_normals_ideal()
            test_targets = data.load_images(
                cfg.TEST.IMAGES.PATHS,
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

    @torch.no_grad()
    def season_plot(
            self,
            H: AbstractHeliostat,
            epoch: int,
            logdir: str,
    ) -> None:
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
        plt.savefig(os.path.join(logdir, f"season_test_{epoch}"), dpi=fig.dpi)
        plt.close(fig)

    @torch.no_grad()
    def real_data_plot(
            self,
            H: AbstractHeliostat,
            epoch: int,
            logdir: str,
            H_target: AbstractHeliostat,
    ) -> None:
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
            # v_max = th.max(th.stack((ground_truth_train[i]**hg_l,ideal_train[i]**hg_l, prediction_train[i]**hg_l)))
            ax.imshow(ground_truth_train[i]**hg_l, cmap=colormap)
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
            ax.imshow(ideal_train[i]**hg_l, cmap=colormap)
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
            ax.imshow(deflec_train[i]**hg_l, cmap=colormap)
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
            ax.imshow(prediction_train[i]**hg_l, cmap=colormap)
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
            # v_max = th.max(th.stack((ground_truth_test[i]**hg_l,ideal_test[i]**hg_l, prediction_test[i]**hg_l)))
            ax.imshow(ground_truth_test[i]**hg_l, cmap=colormap)
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
            ax.imshow(ideal_test[i]**hg_l, cmap=colormap)
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
            ax.imshow(deflec_test[i]**hg_l, cmap=colormap)
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
            ax.imshow(prediction_test[i]**hg_l, cmap=colormap)
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
        plt.savefig(os.path.join(logdir, f"real_data_{epoch}"), dpi=fig.dpi)
        plt.close(fig)

    @torch.no_grad()
    def create_plots(
            self,
            H: AbstractHeliostat,
            epoch: int,
            logdir: str,
            H_target: AbstractHeliostat,
    ) -> None:
        if self.plot_season:
            self.season_plot(H, epoch, logdir)
        if self.plot_real_data:
            self.real_data_plot(H, epoch, logdir, H_target)


@th.no_grad()
def plot_surfaces_mrad(
        heliostat_target: AbstractHeliostat,
        heliostat_pred: AbstractHeliostat,
        epoch: int,
        logdir_surfaces: str,
        writer: Optional[SummaryWriter] = None,
) -> None:
    logdir_mrad = os.path.join(logdir_surfaces, "mrad")
    os.makedirs(logdir_surfaces, exist_ok=True)
    os.makedirs(logdir_mrad, exist_ok=True)

    target_normal_vecs = heliostat_target.normals
    ideal_normal_vecs = heliostat_target._normals_ideal
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

    if writer:
        writer.add_scalar(
            "test/normal_diffs",
            th.sum(diff_angles) / len(diff_angles),
            epoch,
        )

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

    fig = plt.figure(figsize=(15, 6))

    gs = GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1.0, 0.05])

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.scatter(target[:, 0], target[:, 1], c=target[:, 2], cmap="magma")
    ax1.set_xlim(th.min(target[:, 0]), th.max(target[:, 0]))
    ax1.set_ylim(th.min(target[:, 1]), th.max(target[:, 1]))
    ax1.title.set_text('Original Surface [mrad]')
    ax1.set_aspect("equal")
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.scatter(pred[:, 0], pred[:, 1], c=pred[:, 2], cmap="magma")
    ax2.set_xlim(th.min(pred[:, 0]), th.max(pred[:, 0]))
    ax2.set_ylim(th.min(pred[:, 1]), th.max(pred[:, 1]))
    ax2.title.set_text('Predicted Surface [mrad]')
    ax2.set_aspect("equal")
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.scatter(diff[:, 0], diff[:, 1], c=diff[:, 2], cmap="magma")
    ax3.set_xlim(th.min(diff[:, 0]), th.max(diff[:, 0]))
    ax3.set_ylim(th.min(diff[:, 1]), th.max(diff[:, 1]))
    ax3.title.set_text('Difference [mrad]')
    ax3.set_aspect("equal")
    ax3.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])

    # ax4 = fig.add_subplot(gs[1, 0:2])
    ax4 = plt.subplot(gs[1, 0:2])
    plt.colorbar(im1, orientation='horizontal', cax=ax4, format='%.0e')

    ax5 = plt.subplot(gs[1, 2])
    plt.colorbar(im3, orientation='horizontal', cax=ax5, format='%.0e')
    plt.tight_layout()

    fig.savefig(os.path.join(logdir_mrad, f"test_{epoch}"))
    plt.close(fig)


@th.no_grad()
def plot_surfaces_mm(
        heliostat_target: AbstractHeliostat,
        heliostat_pred: AbstractHeliostat,
        epoch: int,
        logdir_surfaces: str,
        writer: Optional[SummaryWriter] = None,
) -> None:
    logdir_mm = os.path.join(logdir_surfaces, "mm")
    os.makedirs(logdir_surfaces, exist_ok=True)
    os.makedirs(logdir_mm, exist_ok=True)

    target = heliostat_target.discrete_points
    ideal = heliostat_target._discrete_points_ideal
    target = target.detach().cpu()
    ideal = ideal.detach().cpu()

    # print(target.shape)
    # print(ideal.shape)
    target[:, -1] = target[:, -1] - ideal[:, -1]

    # target[:, 2] = target[:, 2]  # / 1e-3

    pred = heliostat_pred.discrete_points
    pred = pred.detach().cpu()
    pred[:, -1] = pred[:, -1] - ideal[:, -1]

    # pred[:, 2] = pred[:, 2]  # / 1e-3

    diff = pred.clone()
    diff[:, 2] = pred[:, 2] - target[:, 2]  # / 10e-3
    if writer:
        writer.add_scalar(
            "test/location_diffs",
            th.sum(abs(diff[:, 2])) / len(diff[:, 2]),
            epoch,
        )

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))
    plt.subplots_adjust(left=0.03, top=0.95, right=0.97, bottom=0.15)

    p0 = ax1.get_position().get_points().flatten()
    p1 = ax2.get_position().get_points().flatten()
    p2 = ax3.get_position().get_points().flatten()
    ax_cbar = fig.add_axes([p0[0], 0.05, p1[2] - p0[0], 0.05])
    ax_cbar1 = fig.add_axes([p2[0], 0.05, p2[2] - p2[0], 0.05])

    im1 = ax1.scatter(target[:, 0], target[:, 1], c=target[:, 2])
    ax1.set_xlim(th.min(target[:, 0]), th.max(target[:, 0]))
    ax1.set_ylim(th.min(target[:, 1]), th.max(target[:, 1]))
    ax1.title.set_text('Original Surface [mm]')
    ax1.set_aspect("equal")

    im2 = ax2.scatter(pred[:, 0], pred[:, 1], c=pred[:, 2])
    ax2.set_xlim(th.min(pred[:, 0]), th.max(pred[:, 0]))
    ax2.set_ylim(th.min(pred[:, 1]), th.max(pred[:, 1]))
    ax2.title.set_text('Predicted Surface [mm]')
    ax2.set_aspect("equal")

    im3 = ax3.scatter(diff[:, 0], diff[:, 1], c=diff[:, 2], cmap="magma")
    ax3.set_xlim(th.min(diff[:, 0]), th.max(diff[:, 0]))
    ax3.set_ylim(th.min(diff[:, 1]), th.max(diff[:, 1]))
    ax3.title.set_text('Difference [mm]')
    ax3.set_aspect("equal")

    plt.colorbar(im1, cax=ax_cbar, orientation='horizontal', format='%.0e')
    plt.colorbar(im3, cax=ax_cbar1, orientation='horizontal', format='%.0e')

    fig.savefig(os.path.join(logdir_mm, f"test_{epoch}"))
    plt.close(fig)


def target_image_comparision_pred_orig_naive(
        ae: torch.Tensor,
        original: torch.Tensor,
        predicted: torch.Tensor,
        naive: torch.Tensor,
        train_sun_position: torch.Tensor,
        epoch: int,
        logdir: str,
        start_main_plot_at_row: int = 1,
) -> None:
    num_azi = len(th.unique(ae[:, 0]))
    num_ele = len(th.unique(ae[:, 1]))

    ae = ae.detach().cpu()
    train_sun_position = train_sun_position.detach().cpu()

    small_width = [0.2] * num_ele * 4
    width_ratios = [1.0] * num_ele * 4
    width_ratios[3::4] = small_width[3::4]

    column = num_azi
    row = num_ele

    height_ratios = [1] * (num_azi + start_main_plot_at_row)
    height_ratios[0] = 2

    loss_fn = th.nn.L1Loss()
    # TODO `row` and `column` seem to be swapped in meaning.
    row = num_ele * 4
    column = num_azi
    fig, axs = plt.subplots(
        column + 1,
        row,
        figsize=(5 * num_ele, 2 * num_azi),
        sharex=True,
        sharey=True,
        gridspec_kw={
            'width_ratios': width_ratios,
            'height_ratios': height_ratios,
        },
    )
    gs = axs[1, 1].get_gridspec()

    j = 0

    original = original.detach().cpu()
    predicted = predicted.detach().cpu()
    naive = naive.detach().cpu()

    smp = start_main_plot_at_row * row  # start main plot
    for (i, ax) in enumerate(axs.flat):
        # Nested Subplots
        # if i == 0:
        #     ax.remove()

        if i < smp:
            ax.remove()
        # Modifications for all Plots
        if i >= smp:
            ax.set_aspect('equal')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        # Modification for each row

        if i % row == 0 and i >= smp:
            ax.set_ylabel(
                "Azimuth = " + str(int(ae[j, 0])),
                fontweight='bold',
                fontsize=12,
            )
            # ax.set_ylabel("Azimuth = " + str(int(ae[j, 0])))

        # Modification specific for each plot

        output_pred = loss_fn(predicted[j], original[j])
        output_naive = loss_fn(naive[j], original[j])

        if i % 4 == 0 and i >= smp:
            ax.imshow(predicted[j], cmap="coolwarm")

            if output_pred < output_naive:
                font = "bold"
            else:
                font = "normal"
            ax.set_xlabel(
                "L1: " + f"{output_pred.item():.4f}", fontweight=font)
            if i - smp < row:
                ax.set_title('Predicted', fontweight='bold')

        elif i % 4 == 1 and i >= smp:
            ax.imshow(original[j], cmap="coolwarm")
            if i - smp < row:
                ax.set_title('Original', fontweight='bold')
            if i - smp >= row * (column - 1):
                ax.set_xlabel(
                    "Elevation = " + str(int(ae[j, 1])),
                    fontweight='bold',
                    fontsize=12,
                )

        elif i % 4 == 2 and i >= smp:
            ax.imshow(naive[j], cmap="coolwarm")

            if output_naive < output_pred:
                font = "bold"
            else:
                font = "normal"

            ax.set_xlabel(
                "L1: " + f"{output_naive.item():.4f}", fontweight=font)
            if i - smp < row:
                ax.set_title('Naive', fontweight='bold')
        elif i % 4 == 3 and i >= smp:
            ax.remove()

        if i % 4 == 3 and i >= smp:
            j += 1

    axbig = fig.add_subplot(gs[0:1, 4:8], projection='polar')

    axbig.set_thetamin(-90)

    axbig.set_thetamax(90)

    axbig.set_theta_zero_location("N")

    axbig.set_rorigin(-95)

    axbig.scatter(
        th.deg2rad(ae[:, 0]),
        -ae[:, 1],
        color='r',
        marker='x',
        s=10,
        label="Test sun positions",
    )

    train_sun_position = utils.vec_to_ae(train_sun_position)
    axbig.scatter(
        th.deg2rad(train_sun_position[:, 0]),
        -train_sun_position[:, 1],
        color='b',
        marker='x',
        s=10,
        label="Train sun position",
    )
    axbig.legend(loc='upper right', bbox_to_anchor=(-0.1, 0.5, 0.5, 0.5))
    axbig.set_yticks(th.arange(-90, 20, 30))

    axbig.set_yticklabels(abs(axbig.get_yticks()))
    axbig.set_ylabel('Azimuth', rotation=67.5)
    axbig.set_xlabel('Elevation')

    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, f"enhanced_test_{epoch}"))
    plt.close(fig)


def spherical_loss_plot(
        train_vec: torch.Tensor,
        spheric_ae: torch.Tensor,
        train_loss: torch.Tensor,
        spheric_losses: torch.Tensor,
        naive_losses: torch.Tensor,
        num_spheric_samples: int,
        epoch: int,
        logdir: str,
) -> None:
    """
    spheric_ae and losses are seperated in 3 parts
    (`nums` is `num_spheric_samples`)
    [:nums] = constant elevation and western azimuth hemisphere
              (from train vector viewing position)
    [nums:2*nums] = constant elevation and eastern hemisphere
    [2*nums:] = constant azimuth with all possible elevations
    """
    # To CPU
    train_vec = train_vec.detach().cpu()
    train_loss = train_loss.detach().cpu()
    spheric_losses = spheric_losses.detach().cpu()
    naive_losses = naive_losses.detach().cpu()
    # Radial Plot Calculations
    train_ae = utils.vec_to_ae(train_vec)
    ae = spheric_ae.clone()

    # Setup Figure
    height_ratios = [1.0, 0.3]
    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=height_ratios)

    # Fill First Plot
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    ax1.set_theta_zero_location("N")

    nums = num_spheric_samples
    l1 = ax1.scatter(
        th.deg2rad(ae[:nums, 0]),
        -ae[:nums, 1],
        marker='.',
        s=40,
        label="Test: Left-handed azimuth angles",
    )
    l2 = ax1.scatter(
        th.deg2rad(ae[nums:2 * nums, 0]),
        -ae[nums:2 * nums, 1],
        marker='.',
        s=40,
        label="Test: Right-handed azimuth angles",
    )
    l3 = ax1.scatter(
        th.deg2rad(ae[2 * nums:, 0]),
        -ae[2 * nums:, 1],
        marker='.',
        s=40,
        label="Test: Elevation angles",
    )
    l4 = ax1.scatter(
        th.deg2rad(train_ae[:, 0]),
        -train_ae[:, 1],
        marker='*',
        s=40,
        label="Azi./Ele. of training",
    )

    # Axis Ticks
    ax1.set_yticks(th.arange(-90, -10, 20))
    ax1.set_yticklabels(abs(ax1.get_yticks()))
    ax1.set_rlabel_position(0)
    tick_labels = ["0", "45", "90", "135", "$\\pm$ 180", "-135", "-90", "-45"]
    value_list = ax1.get_xticks().tolist()
    ax1.xaxis.set_ticks(value_list)
    ax1.set_xticklabels(tick_labels)
    # Axis Labels
    ax1.set_xlabel(r'Azimuth $\theta^{a}$ [°]')
    label_position = ax1.get_rlabel_position()
    ax1.text(
        np.deg2rad(label_position + 7),
        -63,
        r'Elevation $\theta^{e}$[°]',
        rotation=91,
        ha='center',
        va='center',
    )

    # Calculations For Second Plot
    # predictions
    ae[:nums, 0] = ae[:nums, 0] - train_ae[0, 0]
    azi_west_no_offsets = th.abs(
        th.where(
            ae[:nums, 0] > 0,
            ae[:nums, 0],
            360 + ae[:nums, 0],
        ) - 360  # delay to zero
    )
    azi_west_loss = spheric_losses[:nums]  # same for y values

    ae[nums:2 * nums, 0] = ae[nums:2 * nums, 0] - train_ae[0, 0]
    azi_east_no_offsets = th.where(
        ae[nums:2 * nums, 0] > 0,
        ae[nums:2 * nums, 0],
        360 + ae[nums:2 * nums, 0],
    ) % 360  # delay to 0–180
    azi_east_loss = spheric_losses[nums:2 * nums]  # same for y values

    ele_no_offsets = th.where(
        ae[2 * nums:, 0] < 0,
        ae[2 * nums:, 1] - train_ae[0, 1],
        180 - ae[2 * nums:, 1] - train_ae[0, 1],
    )  # delay to 180–0
    ele_loss = spheric_losses[2 * nums:]

    # naive
    naive_azi_west_loss = naive_losses[:nums]  # same for y values
    naive_azi_east_loss = naive_losses[nums:2 * nums]  # same for y values
    naive_ele_loss = naive_losses[2 * nums:]

    # Fill Second Figure
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.plot(azi_west_no_offsets, azi_west_loss, marker='.', zorder=3)
    ax2.plot(azi_east_no_offsets, azi_east_loss, marker='.', zorder=6)
    ax2.plot(ele_no_offsets, ele_loss, marker='.', zorder=8)
    ax2.scatter(0, train_loss, s=90, marker='*', color="r", zorder=10)

    l1_naive = ax2.plot(
        azi_west_no_offsets,
        naive_azi_west_loss,
        zorder=3,
        color="cornflowerblue",
        label="Naive loss left-azi",
    )
    l2_naive = ax2.plot(
        azi_east_no_offsets,
        naive_azi_east_loss,
        zorder=6,
        color="bisque",
        label="Naive loss right-azi",
    )
    l3_naive = ax2.plot(
        ele_no_offsets,
        naive_ele_loss,
        zorder=8,
        color="limegreen",
        label="Naive loss ele",
    )

    # ax2.set_xlim(-20, 180)

    # Axis Labels
    ax2.set_xlabel(r'$|\theta^{a,e}_{test}|-\theta^{a,e}_{train}$ [°]')
    ax2.set_ylabel('L1 Loss')
    # ax2.set_ylim(0, 12)

    # Legend
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    ax3.legend(
        handles=[
            l1,
            l1_naive[0],
            l2,
            l2_naive[0],
            l3,
            l3_naive[0],
            l4,
        ],
        loc="center",
    )

    # plt.tight_layout()
    plt.savefig(os.path.join(logdir, f"spheric_test_{epoch}"))
    plt.close(fig)


def season_plot_nature(
        season_extras: Dict[str, Any],
        ideal: torch.Tensor,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        prediction_loss: torch.Tensor,
        ground_truth_loss: torch.Tensor,
        epoch: int,
        logdir: str,
) -> None:
    colormap = "hot"
    faktor = 1.3
    fig = plt.figure(figsize=(27, 8))
    width_ratios = [1, 1, 1, 0.1, 1, 1, 1, 0.1, 1, 1, 1, 0.1, 1, 1, 1]

    # fig = plt.figure(constrained_layout=True, figsize=(27, 10))

    gs = GridSpec(
        3,
        15,
        figure=fig,
        width_ratios=width_ratios,
        wspace=0,
        hspace=0
    )

    # fig.set_constrained_layout_pads(w_pad=0, h_pad=0)
    font_pred = []
    font_gt = []
    for i in range(len(prediction_loss)):
        if prediction_loss[i] < ground_truth_loss[i]:
            font_pred.append("bold")
            font_gt.append("normal")
        else:
            font_gt.append("bold")
            font_pred.append("normal")

    # gs = GridSpec(3, 16, width_ratios, figure=fig)
    # width_ratios=[1, 1, 1], height_ratios=[1.0, 0.05])

    # plt.subplots_adjust(wspace=0)
    # 9,12,15,18
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(ground_truth[0]**faktor, cmap=colormap)
    ax00.axis('off')
    ax00.set_title("Ground Truth", size=18)
    string = 'Shortest Day'
    ax00.text(
        -0.12,
        0.5,
        string,
        horizontalalignment='center',
        verticalalignment='center',
        rotation=90,
        size=18,
        transform=ax00.transAxes,
    )

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.imshow(ideal[0]**faktor, cmap=colormap)
    ax01.axis('off')
    ax01.set_title("Ideal", size=18)
    ax01.text(
        0.5,
        -0.10,
        f"L1: {ground_truth_loss[0].item():.4f}",
        size=18,
        ha="center",
        transform=ax01.transAxes,
        fontweight=font_gt[0],
    )
    ax01.text(
        0.5,
        1.2,
        "9:00 a.m.",
        size=18,
        ha="center",
        transform=ax01.transAxes,
    )

    ax02 = fig.add_subplot(gs[0, 2])
    ax02.imshow(prediction[0]**faktor, cmap=colormap)
    ax02.axis('off')
    ax02.set_title("Prediction", size=18)
    ax02.text(
        0.5,
        -0.10,
        f"L1: {prediction_loss[0].item():.4f}",
        size=18,
        ha="center",
        transform=ax02.transAxes,
        fontweight=font_pred[0],
    )

    ax04 = fig.add_subplot(gs[0, 4])
    ax04.imshow(ground_truth[1]**faktor, cmap=colormap)
    ax04.axis('off')
    ax04.set_title("Ground Truth", size=18)

    ax05 = fig.add_subplot(gs[0, 5])
    ax05.imshow(ideal[1]**faktor, cmap=colormap)
    ax05.axis('off')
    ax05.set_title("Ideal", size=18)
    ax05.text(
        0.5,
        -0.10,
        f"L1: {ground_truth_loss[1].item():.4f}",
        size=18,
        ha="center",
        transform=ax05.transAxes,
    )
    ax05.text(
        0.5,
        1.2,
        "12:00 p.m.",
        size=18,
        ha="center",
        transform=ax05.transAxes,
        fontweight=font_gt[1],
    )

    ax06 = fig.add_subplot(gs[0, 6])
    ax06.imshow(prediction[1]**faktor, cmap=colormap)
    ax06.axis('off')
    ax06.set_title("Prediction", size=18)
    ax06.text(
        0.5,
        -0.10,
        f"L1: {prediction_loss[1].item():.4f}",
        size=18,
        ha="center",
        transform=ax06.transAxes, fontweight=font_pred[1],
    )

    ax08 = fig.add_subplot(gs[0, 8])
    ax08.imshow(th.zeros_like(prediction[1]), cmap="Greys")
    ax08.axis('off')
    ax08.set_title("Ground Truth", size=18)

    ax09 = fig.add_subplot(gs[0, 9])
    ax09.imshow(th.zeros_like(prediction[1]), cmap="Greys")
    ax09.axis('off')
    ax09.set_title("Ideal", size=18)
    ax09.text(
        0.5,
        1.2,
        "03:00 p.m.",
        size=18,
        ha="center",
        transform=ax09.transAxes,
    )

    ax010 = fig.add_subplot(gs[0, 10])
    ax010.imshow(th.zeros_like(prediction[1]), cmap="Greys")
    ax010.axis('off')
    ax010.set_title("Prediction", size=18)

    ax012 = fig.add_subplot(gs[0, 12])
    ax012.imshow(th.zeros_like(prediction[1]), cmap="Greys")
    ax012.axis('off')
    ax012.set_title("Ground Truth", size=18)

    ax013 = fig.add_subplot(gs[0, 13])
    ax013.imshow(th.zeros_like(prediction[1]), cmap="Greys")
    ax013.axis('off')
    ax013.set_title("Ideal", size=18)
    ax013.text(
        0.5,
        1.2,
        "6:00 p.m.",
        size=18,
        ha="center",
        transform=ax013.transAxes,
    )

    ax014 = fig.add_subplot(gs[0, 14])
    ax014.imshow(th.zeros_like(prediction[1]), cmap="Greys")
    ax014.axis('off')
    ax014.set_title("Prediction", size=18)

    ax10 = fig.add_subplot(gs[1, 0])
    ax10.imshow(ground_truth[2]**faktor, cmap=colormap)
    string = 'Equinox'
    ax10.text(
        -0.15,
        0.5,
        string,
        horizontalalignment='center',
        verticalalignment='center',
        rotation=90,
        size=18,
        transform=ax10.transAxes,
    )
    ax10.axis('off')

    ax11 = fig.add_subplot(gs[1, 1])
    ax11.imshow(ideal[2]**faktor, cmap=colormap)
    ax11.axis('off')
    ax11.text(
        0.5,
        -0.10,
        f"L1: {ground_truth_loss[2].item():.4f}",
        size=18,
        ha="center",
        transform=ax11.transAxes,
        fontweight=font_gt[2],
    )

    ax12 = fig.add_subplot(gs[1, 2])
    ax12.imshow(prediction[2]**faktor, cmap=colormap)
    ax12.axis('off')
    ax12.text(
        0.5,
        -0.10,
        f"L1: {prediction_loss[2].item():.4f}",
        size=18,
        ha="center",
        transform=ax12.transAxes, fontweight=font_pred[2],
    )

    ax14 = fig.add_subplot(gs[1, 4])
    ax14.imshow(ground_truth[3]**faktor, cmap=colormap)
    ax14.axis('off')

    ax15 = fig.add_subplot(gs[1, 5])
    ax15.imshow(ideal[3]**faktor, cmap=colormap)
    ax15.axis('off')
    ax15.text(
        0.5,
        -0.10,
        f"L1: {ground_truth_loss[3].item():.4f}",
        size=18,
        ha="center",
        transform=ax15.transAxes,
        fontweight=font_gt[3],
    )

    ax16 = fig.add_subplot(gs[1, 6])
    ax16.imshow(prediction[3]**faktor, cmap=colormap)
    ax16.axis('off')
    ax16.text(
        0.5,
        -0.10,
        f"L1: {prediction_loss[3].item():.4f}",
        size=18,
        ha="center",
        transform=ax16.transAxes, fontweight=font_pred[3],
    )

    ax18 = fig.add_subplot(gs[1, 8])
    ax18.imshow(ground_truth[4]**faktor, cmap=colormap)
    ax18.axis('off')

    ax19 = fig.add_subplot(gs[1, 9])
    ax19.imshow(ideal[4]**faktor, cmap=colormap)
    ax19.axis('off')
    ax19.text(
        0.5,
        -0.10,
        f"L1: {ground_truth_loss[4].item():.4f}",
        size=18,
        ha="center",
        transform=ax19.transAxes,
        fontweight=font_gt[4],
    )

    ax110 = fig.add_subplot(gs[1, 10])
    ax110.imshow(prediction[4]**faktor, cmap=colormap)
    ax110.axis('off')
    ax110.text(
        0.5,
        -0.10,
        f"L1: {prediction_loss[4].item():.4f}",
        size=18,
        ha="center",
        transform=ax110.transAxes,
        fontweight=font_pred[4],
    )

    ax20 = fig.add_subplot(gs[2, 0])
    ax20.imshow(ground_truth[5]**faktor, cmap=colormap)
    string = 'Longest Day'
    ax20.text(
        -0.15,
        0.5,
        string,
        horizontalalignment='center',
        verticalalignment='center',
        rotation=90,
        size=18,
        transform=ax20.transAxes,
    )
    ax20.axis('off')

    ax21 = fig.add_subplot(gs[2, 1])
    ax21.imshow(ideal[5]**faktor, cmap=colormap)
    ax21.axis('off')
    ax21.text(
        0.5,
        -0.10,
        f"L1: {ground_truth_loss[5].item():.4f}",
        size=18,
        ha="center",
        transform=ax21.transAxes,
        fontweight=font_gt[5],
    )

    ax22 = fig.add_subplot(gs[2, 2])
    ax22.imshow(prediction[5]**faktor, cmap=colormap)
    ax22.axis('off')
    ax22.text(
        0.5,
        -0.10,
        f"L1: {prediction_loss[5].item():.4f}",
        size=18,
        ha="center",
        transform=ax22.transAxes,
        fontweight=font_pred[5],
    )

    ax24 = fig.add_subplot(gs[2, 4])
    ax24.imshow(ground_truth[6]**faktor, cmap=colormap)
    ax24.axis('off')

    ax25 = fig.add_subplot(gs[2, 5])
    ax25.imshow(ideal[6]**faktor, cmap=colormap)
    ax25.axis('off')
    ax25.text(
        0.5,
        -0.10,
        f"L1: {ground_truth_loss[6].item():.4f}",
        size=18,
        ha="center",
        transform=ax25.transAxes,
        fontweight=font_gt[6],
    )

    ax26 = fig.add_subplot(gs[2, 6])
    ax26.imshow(prediction[6]**faktor, cmap=colormap)
    ax26.axis('off')
    ax26.text(
        0.5,
        -0.10,
        f"L1: {prediction_loss[6].item():.4f}",
        size=18,
        ha="center",
        transform=ax26.transAxes,
        fontweight=font_pred[6],
    )

    ax28 = fig.add_subplot(gs[2, 8])
    ax28.imshow(ground_truth[7]**faktor, cmap=colormap)
    ax28.axis('off')

    ax29 = fig.add_subplot(gs[2, 9])
    ax29.imshow(ideal[7]**faktor, cmap=colormap)
    ax29.axis('off')
    ax29.text(
        0.5,
        -0.10,
        f"L1: {ground_truth_loss[7].item():.4f}",
        size=18,
        ha="center",
        transform=ax29.transAxes,
        fontweight=font_gt[7],
    )

    ax210 = fig.add_subplot(gs[2, 10])
    ax210.imshow(prediction[7]**faktor, cmap=colormap)
    ax210.axis('off')
    ax210.text(
        0.5,
        -0.10,
        f"L1: {prediction_loss[7].item():.4f}",
        size=18,
        ha="center",
        transform=ax210.transAxes,
        fontweight=font_pred[7],
    )

    ax212 = fig.add_subplot(gs[2, 12])
    ax212.imshow(ground_truth[8]**faktor, cmap=colormap)
    ax212.axis('off')

    ax213 = fig.add_subplot(gs[2, 13])
    ax213.imshow(ideal[8]**faktor, cmap=colormap)
    ax213.axis('off')
    ax213.text(
        0.5,
        -0.10,
        f"L1: {ground_truth_loss[8].item():.4f}",
        size=18,
        ha="center",
        transform=ax213.transAxes,
        fontweight=font_gt[8],
    )

    ax214 = fig.add_subplot(gs[2, 14])
    ax214.imshow(prediction[8]**faktor, cmap=colormap)
    ax214.axis('off')
    ax214.text(
        0.5,
        -0.10,
        f"L1: {prediction_loss[8].item():.4f}",
        size=18,
        ha="center",
        transform=ax214.transAxes,
        fontweight=font_pred[8],
    )

    # plt.show()
    # if i == 1:
    #     string = 'Equinox'+'\n'+' 12:00 a.m.'
    #     ax.text(
    #         -0.15,
    #         0.5,
    #         string,
    #         horizontalalignment='center',
    #         verticalalignment='center',
    #         rotation=90,
    #         size=18,
    #         transform=ax.transAxes,
    #     )
    # if i == 2:
    #     string = 'Longest Day'+'\n'+' 18:00 a.m.'
    #     ax.text(
    #         -0.15,
    #         0.5,
    #         string,
    #         horizontalalignment='center',
    #         verticalalignment='center',
    #         rotation=90,
    #         size=18,
    #         transform=ax.transAxes,
    #     )
    # ax = fig.add_subplot(gs[i, 1])
    # ax.imshow(ideal[i]**1.3, cmap=colormap)
    # ax.axis('off')
    # ax.text(
    #     0.5,
    #     -0.10,
    #     f"L1: {ground_truth_loss[i].item():.4f}",
    #     size=18,
    #     ha="center",
    #     transform=ax.transAxes,
    # )
    # if i == 0:
    #     ax.set_title("Ideal", size=18)
    # ax = fig.add_subplot(gs[i, 2])
    # ax.imshow(prediction[i]**1.3, cmap=colormap)
    # ax.axis('off')
    # ax.text(
    #     0.5,
    #     -0.10,
    #     f"L1: {prediction_loss[i].item():.4f}",
    #     size=18,
    #     ha="center",
    #     transform=ax.transAxes,
    #     fontweight="bold",
    # )
    # if i == 0:
    #     ax.set_title("Prediction", size=18)
    plt.tight_layout()
    # # plt.show()
    plt.savefig(os.path.join(logdir, f"season_test_{epoch}"), dpi=fig.dpi)
    plt.close(fig)



    
    
