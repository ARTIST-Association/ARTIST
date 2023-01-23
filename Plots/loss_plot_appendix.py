# from packaging import version
import sys
sys.path.append('../')
# import pandas as pd
# from matplotlib import pyplot as plt
# # import seaborn as sns
# from scipy import stats
# import tensorboard as tb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
# from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib
from defaults import get_cfg_defaults, load_config_file
from build_heliostat_model import build_target_heliostat, build_heliostat, load_heliostat
import disk_cache
from data import generate_sun_array
from matplotlib.gridspec import GridSpec
import numpy as np
import torch as th
from render import Renderer
from environment import Environment
# def parse_tensorboard(path, scalars):
#     """returns a dictionary of pandas dataframes for each requested scalar"""
#     ea = event_accumulator.EventAccumulator(
#         path,
#         size_guidance={event_accumulator.SCALARS: 0},
#     )
#     _absorb_print = ea.Reload()
#     # make sure the scalars are in the event accumulator tags
#     assert all(
#         s in ea.Tags()["scalars"] for s in scalars
#     ), "some scalars were not found in the event accumulator"
#     return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}
def get_losses(cfg_default, event_files, config_files):
    # print(event_files)
    min_loss_test = []
    min_normal_diffs = []
    tick_labels = []
    best_prediction_image = []
    target_image_list = []
    for event,load_cfg in zip(event_files,config_files):

        cfg = load_config_file(cfg_default, load_cfg)
        cfg.freeze()
        
        if case =="distance":
            tick_labels.append(cfg.H.DEFLECT_DATA.POSITION_ON_FIELD[1])
            x_label = "Horizontal distance to tower / m"
        elif case=="num_images":
            tick_labels.append(cfg.TRAIN.SUN_DIRECTIONS.RAND.NUM_SAMPLES)
            x_label = "# Training Images"
            
            
        
        ea = EventAccumulator(event)
        ea.Reload()
        
        _, epoch, loss_train = zip(*ea.Scalars('train/loss_scaled'))
        _, epoch, loss_test = zip(*ea.Scalars('test/loss_scaled'))
        _, epoch, normal_diffs = zip(*ea.Scalars('test/normal_diffs'))
        prediction_images = zip(*ea.Images('test/prediction_4'))
        target_image = ea.Images('test/target_4')
        

        min_loss_test.append(min(loss_test))
        min_normal_diffs.append(min(normal_diffs))
        target_image_list.append(target_image)
    sorted_test_loss  = [loss * cfg.AC.RECEIVER.PLANE_X * cfg.AC.RECEIVER.PLANE_Y for label, loss in sorted(zip(tick_labels, min_loss_test))]
    sorted_normal_diffs  = [loss/1e-3 for label, loss in sorted(zip(tick_labels, min_normal_diffs))]
    return sorted_test_loss, sorted_normal_diffs, tick_labels, target_image_list

def loss_plot(fig):
    ax_flux = fig.add_subplot()
    plt.yticks(fontsize = 18)
    ax_surface = ax_flux.twinx()
    plt.yticks(fontsize = 18)
    
    linestyle = ["dashdot", "dashed","dotted","solid" ]
    marker     = ["v", "*","^","d" ]
    num        = ["2", "4","8","16"]
    plot= []
    
    if case =="distance":
        x_label = "Horizontal distance to tower / m"
    elif case=="num_images":
        x_label = "# Training Images"
    
    for j, experiments in enumerate(experiment_path_flux_density):
        print(j)
        print(experiments)
        experiment_id = os.path.join(*experiments, 'event*')
        experiment_config = os.path.join(*experiments, 'config.yaml')
        # print(experiment_config)
        event_files=glob.glob(experiment_id)
        # print(event_files)
        # print(events_files)
        config_files= glob.glob(experiment_config)
        
        cfg_default = get_cfg_defaults()
        
        min_loss_test, min_normal_diffs, tick_labels, target_image = get_losses(cfg_default, event_files, config_files)
    
        plot1 = ax_flux.plot(sorted(tick_labels), min_loss_test, label=f"L1-Loss                trained on {num[j]} images", marker=marker[j], markersize=12, color="tab:blue",linestyle=linestyle[j])
        plot += plot1
        # ax_flux.plot(sorted_test_loss)
    
    for i, experiments in enumerate(experiment_path_surfaces):
        print(experiments)
        experiment_id = os.path.join(*experiments, 'event*')
        experiment_config = os.path.join(*experiments, 'config.yaml')
        # print(experiment_config)
        event_files=glob.glob(experiment_id)
        # print(event_files)
        # print(events_files)
        config_files= glob.glob(experiment_config)
        
        cfg_default = get_cfg_defaults()
        
        min_loss_test, min_normal_diffs, tick_labels, _ = get_losses(cfg_default, event_files, config_files)
    
        plot2 = ax_surface.plot(sorted(tick_labels), min_normal_diffs, label=f"Normal deviation trained on {num[i]} images",marker=marker[i], markersize=12, color="tab:orange",linestyle=linestyle[i])
        plot += plot2
    
    # print(tick_labels)
    ax_flux.set_xscale("log")
    ax_flux.set_xticks(sorted(tick_labels))
    ax_flux.set_xticklabels(sorted(tick_labels), size=18)
    # ax_flux.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatter())
    
    ax_surface.set_xticks(tick_labels)
    ax_surface.set_xticklabels(tick_labels, size=18)
    ax_flux.set_xlabel(x_label, size=18)
    ax_flux.set_xlim(24,410)
    # ax_flux.set_ylim(16)
    ax_flux.set_ylabel(r"L1-Test loss $\cdot$ cal. target surface / $\Delta E \cdot m^2$", size=18)
    
    ax_surface.set_ylabel("Mean normal deviations / mrad", size=18)
    
    labels = [l.get_label() for l in plot]
    plt.legend(plot, labels,fontsize=18,bbox_to_anchor=(-0.07,1), loc=0)
    plt.yticks(fontsize = 18)

def load_config(path):
    MODEL_PATH = os.path.join(*[path,'Logfiles', 'MultiNURBSHeliostat.pt'])
    CONFIG_PATH = os.path.join(*[path,'config.yaml'])
    
    cfg_default = get_cfg_defaults()
    print(f"load: {CONFIG_PATH}")
    cfg = load_config_file(cfg_default, CONFIG_PATH)
    cfg.merge_from_list(['CP_PATH', MODEL_PATH])
    cfg.merge_from_list(['H.DEFLECT_DATA.DIRECTORY', os.path.join("..", cfg.H.DEFLECT_DATA.DIRECTORY)])
    # cfg.merge_from_list(['USE_NURBS', False])
    # cfg.merge_from_list(['H.DEFLECT_DATA.TAKE_N_VECTORS', 800])
    cfg.freeze()
    return cfg

def create_heliostats(cfg):
        
        
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

        return H_trained, H_target
    
def get_flux_maps(path_to_worst_pred, sun_direction):
    prediction = []
    target = []
    cfg_default = get_cfg_defaults()
    print(path_to_worst_pred)
    # experiment_config = os.path.join(*experiments, 'config.yaml')
    paths = glob.glob(os.path.join(*path_to_worst_pred))
    for path in paths:
        cfg = load_config(path)
        H_trained, H_target = create_heliostats(cfg)

        ENV = Environment(cfg.AC, "cpu")
        renderer = Renderer(H_trained, ENV)
        sun_direction, ae = generate_sun_array(
            cfg.TRAIN.SUN_DIRECTIONS, 'cpu')
        H_trained_aligned = H_trained.align(sun_direction[0])
        (
            pred_bitmap,
            (_, _, _, _, _, _),
        ) = renderer.render(H_trained_aligned, return_extras=True)
        prediction.append(pred_bitmap)
        
        H_target_aligned = H_target.align(sun_direction)
        (
            target_bitmap,
            (_, _, _, _, _, _),
        ) = renderer.render(H_target_aligned, return_extras=True)
        prediction.append(target_bitmap)
    return prediction,target

def flux_plot(fig,gs,pred,target):
    for i,(p,t) in enumerate(zip(pred,target)):
        ax_target = fig.add_subplot(gs[i, 0])
        ax_target.imshow(t)
        

def main(experiment_path_surfaces, experiment_path_flux_density, sun_direction):
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(10, 10)
    
    pred, target = get_flux_maps(experiment_path_flux_density[0], sun_direction)
    # flux_plot(fig,gs, pred,target)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("loss_plot_appendix.png")
    
if __name__ == '__main__':
    # experiment_paths=['..','Results', 'Distance_Test', '*']
    experiment_path_surfaces= [
    ['..','Results', 'DistanceWithFocus', '_DistanceWithFocusN7S3_D_*_I_2_' ],
    # ['..','Results', 'DistanceWithFocus', '_DistanceWithFocusN7S3_D_*_I_4_' ],
    # ['..','Results', 'DistanceWithFocus', '_DistanceWithFocusN7S3_D_*_I_8_' ],
    # ['..','Results', 'DistanceWithFocus', '_DistanceWithFocusN7S3_D_*_I_16_' ],
    ]
    experiment_path_flux_density= [
    ['..','Results', 'DistanceWithFocus', '_DistanceWithFocusN11S2_D_*_I_2_' ],
    # ['..','Results', 'DistanceWithFocus', '_DistanceWithFocusN11S2_D_*_I_4_' ],
    # ['..','Results', 'DistanceWithFocus', '_DistanceWithFocusN11S2_D_*_I_8_' ],
    # ['..','Results', 'DistanceWithFocus', '_DistanceWithFocusN11S2_D_*_I_16_' ],
    ]
    sun_direction = th.tensor([-0.43719268, 0.7004466, 0.564125])
    main(experiment_path_surfaces, experiment_path_flux_density, sun_direction)
