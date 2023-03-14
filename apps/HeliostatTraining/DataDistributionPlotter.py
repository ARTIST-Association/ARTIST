# system dependencies
import sys
import os
import datetime

import typing
import torch
import json

import matplotlib as mpl
import matplotlib.pyplot as plt

# local dependencies
lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir, 'lib'))
sys.path.append(lib_dir)

from HeliostatTrainingLib.HeliostatDataset import HeliostatDataset

def plotDataPointsOverAngles(ax, data_points, label, color, marker, size = 50, plot_distances: bool = True, alpha=1.0):
     if len(data_points) > 0:
        azim = [dp.sourceAzim(to_deg=True) for dp in data_points.values()]
        elev = [dp.sourceElev(to_deg=True) for dp in data_points.values()]
        distances = [dp.hausdorff_distance() for dp in data_points.values()]

        if plot_distances:
            for az, el, d in zip(azim, elev, distances):
                if d:
                    circle = plt.Circle((az, el), d / 3.14 * 180 , color=color, fill=False, clip_on=True, alpha=0.2)
                    ax.add_patch(circle)

        ax.scatter(azim, elev, c=color, label=label, marker=marker, s=size, alpha=alpha)
        
        return min(azim), max(azim), min(elev), max(elev)
     else:
        return None
     
def plotDataPointsOverTime(ax, data_points, label, color, marker, size = 50, alpha=1.0):
    dp_dates = [datetime.date(year=dp.created_at.year, month=dp.created_at.month, day=dp.created_at.day) for dp in data_points.values()]
    dp_hours = [dp.created_at.hour + dp.created_at.minute / 60 for dp in data_points.values()]

    ax.scatter(dp_dates, dp_hours, c=color, label=label, marker=marker, s=size, alpha=alpha)

def main(dataset_path: str, dataset_config_path: str):

    with open(dataset_config_path, 'r') as file:
            dataset_config = json.load(file)

    dataset : HeliostatDataset = HeliostatDataset(data_points=dataset_path, dataset_config=dataset_config)

    # plot setup
    mpl.rcParams.update({'font.size': 32})
    fig = plt.figure(figsize=(35.4,5), dpi=4000)
    time_ax = fig.add_subplot(1,2,1)
    angle_ax = fig.add_subplot(1,2,2)
    
    # plot data
    plotDataPointsOverTime(ax=time_ax, data_points=dataset.trainingDataset(), label='Trainig', color=[51.0/255.0, 153.0/255, 1.0], marker='s')
    plotDataPointsOverTime(ax=time_ax, data_points=dataset.testingDataset(), label='Validation', color='green', marker='D')
    plotDataPointsOverTime(ax=time_ax, data_points=dataset.evaluationDataset(), label='Testing', color='orange', marker='*', size=150)

    # plotDataPointsOverTime(ax=time_ax, data_points=dataset.trainingDataset(), label='Trainig', color='grey', marker='.')
    # plotDataPointsOverTime(ax=time_ax, data_points=dataset.testingDataset(), label='Validation', color='grey', marker='.')
    # plotDataPointsOverTime(ax=time_ax, data_points=dataset.evaluationDataset(), label='Testing', color='grey', marker='.')

    train_range = plotDataPointsOverAngles(ax=angle_ax, data_points=dataset.trainingDataset(), label='Trainig', color=[51.0/255.0, 153.0/255, 1.0], marker='s')
    valid_range = plotDataPointsOverAngles(ax=angle_ax, data_points=dataset.testingDataset(), label='Validation', color='green', marker='D')
    test_range = plotDataPointsOverAngles(ax=angle_ax, data_points=dataset.evaluationDataset(), label='Testing', color='orange', marker='*', size=150)

    # train_min_azim, train_max_azim, train_min_elev, train_max_elev = plotDataPointsOverAngles(ax=angle_ax, data_points=dataset.trainingDataset(), label='Trainig', color='grey', marker='.')
    # valid_min_azim, valid_max_azim, valid_min_elev, valid_max_elev = plotDataPointsOverAngles(ax=angle_ax, data_points=dataset.testingDataset(), label='Validation', color='grey', marker='.')
    # test_min_azim, test_max_azim, test_min_elev, test_max_elev = plotDataPointsOverAngles(ax=angle_ax, data_points=dataset.evaluationDataset(), label='Testing', color='grey', marker='.')

    azim_range = []
    elev_range = []
    if train_range:
        azim_range.append(train_range[0])
        azim_range.append(train_range[1])
        elev_range.append(train_range[2])
        elev_range.append(train_range[3])
    if valid_range:
        azim_range.append(valid_range[0])
        azim_range.append(valid_range[1])
        elev_range.append(valid_range[2])
        elev_range.append(valid_range[3])
    if test_range:
        azim_range.append(test_range[0])
        azim_range.append(test_range[1])
        elev_range.append(test_range[2])
        elev_range.append(test_range[3])
    

    min_azim = min(azim_range)
    max_azim = max(azim_range)

    min_elev = min(elev_range)
    max_elev = max(elev_range)

    # formatting
    angle_ax.set_ylabel('Source Elevation\n[Deg]')
    angle_ax.set_xlabel('Source Azimuth\n[Deg]')
    angle_ax.set_xlim(0.95*min_azim, 1.05*max_azim)
    angle_ax.set_ylim(0, 1.05*max_elev)
    time_ax.set_ylabel('Hour')
    time_ax.set_xlabel('Date')
    time_ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
    time_ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=150))
    # time_ax.legend()

    plt.savefig(os.path.abspath(os.path.join('/Users/moritz/Desktop', str(datetime.datetime.now())+ '.pdf')), bbox_inches='tight')

if __name__ == '__main__':
    data_dir = '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Test/25_02_2023_15_31'
    main(dataset_path=os.path.abspath(os.path.join(data_dir, 'dataset_data.csv')),
         dataset_config_path=os.path.abspath(os.path.join(data_dir, 'dataset_config.json')),
         )