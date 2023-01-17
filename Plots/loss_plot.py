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
from defaults import get_cfg_defaults, load_config_file
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


def main(experiment_id, case):
    
    experiment_id = os.path.join(*experiment_paths, 'event*')
    experiment_config = os.path.join(*experiment_paths, 'config.yaml')
    print(experiment_config)
    events_files=glob.glob(experiment_id)
    # print(events_files)
    config_files= glob.glob(experiment_config)
    
    cfg_default = get_cfg_defaults()
    
    min_loss_train = []
    min_loss_test = []
    tick_labels = []
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot()
    for event,load_cfg in zip(events_files,config_files):

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
        
        histograms = ea.Tags()['scalars']
        _, epoch, loss_train = zip(*ea.Scalars('train/loss'))
        _, epoch, loss_test = zip(*ea.Scalars('test/loss'))
        
        min_loss_train.append(min(loss_train)/9)
        min_loss_test.append(min(loss_test)/9)
    sorted_train_loss = [loss for label, loss in sorted(zip(tick_labels, min_loss_train))]
    sorted_test_loss  = [loss for label, loss in sorted(zip(tick_labels, min_loss_test))]
    ax.plot(sorted(tick_labels), sorted_train_loss, label="Train loss", marker="*", markersize=12,)
    ax.plot(sorted(tick_labels), sorted_test_loss, label="Test Loss", marker="^", markersize=12,)
    ax.set_xscale("log")
    # ax.plot(sorted_test_loss)
    ax.set_xticks(tick_labels)
    ax.set_xticklabels(tick_labels, size=18)
    ax.set_xlabel(x_label, size=18)
    plt.yticks(fontsize = 18)
    ax.set_xlim(1,16)
    # ax.set_ylim(16)
    ax.set_ylabel("L1-Loss", size=18)
    ax.legend(fontsize=18)
    plt.tight_layout()
    # ax.x_
    plt.savefig("loss_plot.png")
    
if __name__ == '__main__':
    # experiment_paths=['..','Results', 'Distance_Test', '*']
    experiment_paths= ['..','Results', 'Full_Nature_Sweep', '_Full_Nature_Sweep_*SCH_Cyclic_I_*_N_11_SD_3']
    case = "num_images"
    main(experiment_paths, case)
