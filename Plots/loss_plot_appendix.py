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
import numpy as np
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
    min_loss_test = []
    min_normal_diffs = []
    tick_labels = []
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
        
        _, epoch, loss_train = zip(*ea.Scalars('train/loss'))
        _, epoch, loss_test = zip(*ea.Scalars('test/loss'))
        _, epoch, normal_diffs = zip(*ea.Scalars('test/normal_diffs'))
        
        if cfg.H.DEFLECT_DATA.POSITION_ON_FIELD[1] == 400:
            min_loss_test.append(min(loss_test))
        else:
            min_loss_test.append(min(loss_test)/9)
        min_normal_diffs.append(min(normal_diffs))
        sorted_test_loss  = [loss * cfg.AC.RECEIVER.PLANE_X * cfg.AC.RECEIVER.PLANE_Y for label, loss in sorted(zip(tick_labels, min_loss_test))]
        sorted_normal_diffs  = [loss/1e-3 for label, loss in sorted(zip(tick_labels, min_normal_diffs))]
    return sorted_test_loss, sorted_normal_diffs, tick_labels

def main(experiment_paths, case):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot()
    ax2 = ax.twinx()

    marker = ["*", "v", "s"]
    num = ["2","4","8"]
    plot= []
    
    if case =="distance":
        x_label = "Horizontal distance to tower / m"
    elif case=="num_images":
        x_label = "# Training Images"
    
    for i, experiments in enumerate(experiment_paths):
        # print(experiments)
        experiment_id = os.path.join(*experiments, 'event*')
        experiment_config = os.path.join(*experiments, 'config.yaml')
        # print(experiment_config)
        event_files=glob.glob(experiment_id)
        # print(events_files)
        config_files= glob.glob(experiment_config)
        
        cfg_default = get_cfg_defaults()
        
        min_loss_test, min_normal_diffs, tick_labels = get_losses(cfg_default, event_files, config_files)

        plot1 = ax.plot(sorted(tick_labels), min_loss_test, label=f"Trained on {num[i]} images", marker=marker[i], markersize=12, color="tab:blue")

        # ax.plot(sorted_test_loss)


        

        plot2 = ax2.plot(sorted(tick_labels), min_normal_diffs, label=f"Normal deviation on {num[i]}",marker=marker[i], markersize=12, color="tab:orange")
        plot += plot1 + plot2
    
    # print(tick_labels)
    ax.set_xscale("log")
    ax.set_xticks(sorted(tick_labels))
    ax.set_xticklabels(sorted(tick_labels), size=18)
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatter())

    ax2.set_xticks(tick_labels)
    ax2.set_xticklabels(tick_labels, size=18)
    ax.set_xlabel(x_label, size=18)
    ax.set_xlim(50,400)
    # ax.set_ylim(16)
    ax.set_ylabel("L1-Test loss", size=18)
    plt.yticks(fontsize = 18)
    
    ax2.set_ylabel("Normal deviations [mrad]", size=18)

    labels = [l.get_label() for l in plot]
    plt.legend(plot, labels, loc=0,fontsize=18)
    plt.yticks(fontsize = 18)
    plt.tight_layout()

    plt.savefig("loss_plot_appendix.png")
    
if __name__ == '__main__':
    # experiment_paths=['..','Results', 'Distance_Test', '*']
    experiment_paths= [
    ['..','Results', 'Distance_Nature_Sweep', '_Distance_Nature_Sweep_D_*_I_2_'],
    ['..','Results', 'Distance_Nature_Sweep', '_Distance_Nature_Sweep_D_*_I_4_'],
    ['..','Results', 'Distance_Nature_Sweep', '_Distance_Nature_Sweep_D_*_I_8_']
    ]
    case = "distance"
    main(experiment_paths, case)
