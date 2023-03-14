# system dependencies
import sys
import os
import datetime

import typing
import torch
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pvlib
from pvlib.location import Location
import pandas as pd
# from scipy.stats import skewnorm

# local dependencies
lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir, 'lib'))
sys.path.append(lib_dir)

from HeliostatTrainingLib.HeliostatDataset import HeliostatDataset
from HeliostatTrainingLib.HausdorffMetric import HausdorffMetric

def plotDataPointsOverAngles(ax, data_points, min_created_at, max_created_at, label, color, marker, size = 40, plot_distances: bool = True, alpha=1.0, threshold=0.25, dataset = None): # size = 40
     azim = [dp.sourceAzim(to_deg=True) for dp in data_points.values()]
     elev = [dp.sourceElev(to_deg=True) for dp in data_points.values()]

     for key, dp in data_points.items():
        min_dist, neighbors = dp.distanceToDataset(data_points = dataset._data_points,
                                        num_nearest_neighbors=3,
                                        return_extras = True,
                                        )

        for n in neighbors:
            ax.plot([dp.sourceAzim(to_deg=True), dataset._data_points[n].sourceAzim(to_deg=True)], [dp.sourceElev(to_deg=True), dataset._data_points[n].sourceElev(to_deg=True)], color=color, alpha = 0.4)
                
        # circle = plt.Circle((dp.sourceAzim(to_deg=True), dp.sourceElev(to_deg=True)), min_dist / 3.14 * 180 , color=color, fill=False, clip_on=True, alpha=0.4)
        # ax.add_patch(circle)
     
    #  sizes = [ size * (pow(1.1, dp.alignment_deviation() * 1000.0 - 1) - 0.1) for dp in data_points.values()]

     if label == 'Training' and threshold:
        for az, el in zip(azim, elev):
            # circle = plt.Circle((az, el), threshold / 3.14 * 180 , color=color, fill=True, clip_on=True, alpha=0.07)
            # ax.add_patch(circle)
            # ax.scatter(azim, elev, c=color, s=10000, alpha=0.1)
            ax.scatter(azim, elev, c=color, marker='*', s=150, alpha=0.6)
     elif label == 'Validation' and threshold:
        for az, el in zip(azim, elev):
            # circle = plt.Circle((az, el), threshold / 3.14 * 180 , color=color, fill=True, clip_on=True, alpha=0.07)
            # ax.add_patch(circle)
            # ax.scatter(azim, elev, c=color, s=10000, alpha=0.1)
            ax.scatter(azim, elev, c=color, marker='D', s=60, alpha=0.6)
     else:
        # distances = torch.Tensor([dp.hausdorff_distance() for dp in data_points.values()])
        # for az, el, d in zip(azim, elev, distances):
        #     circle = plt.Circle((az, el), d / 3.14 * 180 , color=color, fill=True, clip_on=True, alpha=0.07)
        #     ax.add_patch(circle)

        time_deltas = [(dp.created_at - min_created_at) / (max_created_at - min_created_at) for dp in data_points.values()]

        # cmap_labels = {'Training' : 'Blues', 'Validation' : 'Greens', 'Testing' : 'Oranges'}
        # cmap = plt.get_cmap(cmap_labels[label])
        cmap = plt.get_cmap('winter')
        try:
            sizes = [(size * dp.alignment_deviation() * 1000.0) ** 2 for dp in data_points.values()]
            # ax.scatter(azim, elev, c=cmap(time_deltas), marker='.', s=sizes, alpha=0.4)
            ax.scatter(azim, elev, c=color, marker=marker, s=size, alpha=0.4)
            # ax.scatter(azim, elev, c=color, marker='.', s=size ** 2, alpha=0.4) 
            # ax.scatter(azim, elev, facecolors='none', edgecolors='grey', marker='.', s=size ** 2, alpha=0.4)
            # ax.scatter(azim, elev, facecolors='none', edgecolors='grey', label=label, marker='.', s=size**2, alpha=0.4)

        except:
            ax.scatter(azim, elev, c=color, marker='.', s=size * 2, alpha=0.4)

     ax.scatter(-1,-1, c=color, label=label, s=200, alpha=0.6)

    #  try:
    #     errors = [dp.alignment_deviation() * 1000.0 for dp in data_points.values()]
    #     for az, el, d in zip(azim, elev, errors):
    #         circle1 = plt.Circle((az, el), d * 0.02 / 3.14 * 180 , color=color, fill=True, clip_on=True, alpha=0.4)
    #         circle2 = plt.Circle((az, el), 0.02 / 3.14 * 180 , color='grey', fill=False, clip_on=True, alpha=0.4)
    #         ax.add_patch(circle1)
    #         ax.add_patch(circle2)
    #  except:
    #      for az, el in zip(azim, elev):
    #         circle1 = plt.Circle((az, el), 0.02 / 3.14 * 180 , color=color, fill=True, clip_on=True, alpha=0.4)
    #         ax.add_patch(circle1)

     return min(azim), max(azim), min(elev), max(elev)

def plotDataPointsOverTime(ax, data_points, min_created_at, max_created_at, label, color, marker, size = 40, alpha=1.0):
    dp_dates = [datetime.date(year=dp.created_at.year, month=dp.created_at.month, day=dp.created_at.day) for dp in data_points.values()]
    dp_hours = [dp.created_at.hour + dp.created_at.minute / 60 for dp in data_points.values()]

    if label == 'Training':
        for d, h in zip(dp_dates, dp_hours):
            # circle = plt.Circle((d, h), threshold / 3.14 * 180 , color=color, fill=True, clip_on=True, alpha=0.07)
            # ax.add_patch(circle)
            # ax.scatter(d, h, c=color, s=10000, alpha=0.1)
            ax.scatter(d, h, c=color, marker='*', s=150, alpha=0.6)

    else:
        time_deltas = [(dp.created_at - min_created_at) / (max_created_at - min_created_at) for dp in data_points.values()]

        # cmap_labels = {'Training' : 'Blues', 'Validation' : 'Greens', 'Testing' : 'Oranges'}
        # cmap = plt.get_cmap(cmap_labels[label])
        cmap = plt.get_cmap('winter')
        try:
            sizes = [(size * dp.alignment_deviation() * 1000.0) ** 2 for dp in data_points.values()]
            # ax.scatter(dp_dates, dp_hours, c=cmap(time_deltas), label=label, marker='.', s=sizes, alpha=0.4)
            ax.scatter(dp_dates, dp_hours, c=color, label=label, marker=marker, s=size, alpha=0.4)
            # ax.scatter(dp_dates, dp_hours, c=color, label=label, marker='.', s=size ** 2, alpha=0.4) 
            # ax.scatter(dp_dates, dp_hours, facecolors='none', edgecolors='grey', label=label, marker='.', s=size ** 2, alpha=0.4)
            # ax.scatter(dp_dates, dp_hours, facecolors='none', edgecolors='grey', label=label, marker='.', s=(size * 2) ** 2, alpha=0.4)

        except:
            ax.scatter(dp_dates, dp_hours, c=color, label=label, marker='.', s=size * 2, alpha=0.4)

def plotErrFrequencies_1(ax, data_points, label, color, alpha=0.6, err_width=0.2):
    
    frequencies = {}
    for dp in data_points.values():
        err = dp.alignment_deviation() * 1000.0
        lb = int(err)
        cat = lb
        while cat < err:
            cat += err_width

        if cat in frequencies:
            frequencies[cat] += 1
        else:
            frequencies[cat] = 1

    acc_frequencies = {}
    data_points = dict(sorted(data_points.items(), key=lambda item: item[1].alignment_deviation()))
    ndp = len(data_points)
    acc_frequencies[0] = ndp
    for i, dp in enumerate(data_points.values()):
        acc_frequencies[dp.alignment_deviation() * 1000] = ndp - (i+1)
        
    # ax.scatter(frequencies.keys(), frequencies.values(), label=label, c=color, alpha=alpha, s=100)
    # for cat, freq in frequencies.items():
    #     ax.plot([cat, cat], [0, freq], c=color, alpha=alpha)

    ax.plot(list(acc_frequencies.keys()), list(acc_frequencies.values()), c=color, alpha=alpha, linewidth=4)

def plotErrFrequencies_2(ax, data_points, label, color, alpha=0.6, err_width=0.2):
    
    frequencies = {}
    for dp in data_points.values():
        err = dp.alignment_deviation() * 1000.0
        lb = int(err)
        cat = lb
        while cat < err:
            cat += err_width

        if cat in frequencies:
            frequencies[cat] += 1
        else:
            frequencies[cat] = 1

    ax.scatter(frequencies.keys(), frequencies.values(), label=label, c=color, alpha=alpha, s=100)
    # for cat, freq in frequencies.items():
    #     ax.plot([cat, cat], [0, freq], c=color, alpha=alpha)

def plotErrOverdist(ax, data_points, min_created_at, max_created_at, label, color, alpha=0.6):
    if label=='Training':
        distances = [0 for dp in data_points.values()]
    else:
        distances = [dp.hausdorff_distance() for dp in data_points.values()]
    acc = [dp.alignment_deviation() * 1000 for dp in data_points.values()]

    # all_created_at = [dp.created_at for dp in data_points.values()]
    # min_created_at = min(all_created_at)
    # max_created_at = max(all_created_at)
    time_deltas = [(dp.created_at - min_created_at) / (max_created_at - min_created_at) for dp in data_points.values()]

    # cmap_labels = {'Training' : 'Blues', 'Validation' : 'Greens', 'Testing' : 'Oranges'}
    # cmap = plt.get_cmap(cmap_labels[label])
    cmap = plt.get_cmap('winter')

    marker_labels = {'Training' : '*', 'Validation' : 'D', 'Testing' : '.'}
    s = 100
    if label == 'Testing':
        s *= 2

    if label == 'Validation':
        s *= 0.6
    ax.scatter(acc, distances, c=cmap(time_deltas), alpha=alpha, s=s, marker=marker_labels[label])
    ax.scatter(-1, -1, c=color, alpha=0.6, s = s, label=label, marker=marker_labels[label])

    dist_im = ax.scatter(acc, torch.ones(len(acc)), c=distances, cmap=cmap, s = 0, marker=marker_labels[label])

    mu = torch.mean(torch.Tensor(acc))

    ax.plot([mu, mu], [0, 100], c=color, alpha=0.2, linewidth=5)

    return max(acc), max(distances), dist_im

def nd(x, mu, sig):
    # return (1 / np.sqrt(2 * np.pi * (sig**2))) * np.exp((-1/2) * ((x-mu) / sig)**2)
    return (1 / torch.sqrt(2 * torch.pi * (sig**2))) * torch.exp((-1/2) * (x-mu)**2 / (sig**2))

def plotGauss(ax, ax2, data_points, date_range, label, color, alpha=0.4, dist_width = 0.05, size=80, lat=50.9224226, lng=6.3639119):
    frequencies = {}
    distances = []
    site = Location(latitude=lat, longitude=lng)

    el_list = []
    az_list = []
    dist_list = []

    start_date = datetime.datetime(year=date_range[0].year, month=date_range[0].month, day=date_range[0].day)
    end_date = datetime.datetime(year=date_range[1].year, month=date_range[1].month, day=date_range[1].day)
    date = start_date
    while date <= end_date:
        
        start_hour=datetime.datetime(year=date.year, month=date.month, day=date.day, hour=7)
        end_hour=datetime.datetime(year=date.year, month=date.month, day=date.day, hour=18)
        times = pd.date_range(start=start_hour, end=end_hour, freq='H')
        solpos = site.get_solarposition(times)
        for i in range(len(solpos)):
            az = torch.deg2rad(torch.tensor(solpos.azimuth[i]))
            el = torch.deg2rad(torch.tensor(solpos.elevation[i]))
            dist = HausdorffMetric.distanceToDataset_angles(azim=az, elev=el, data_points=data_points)
            el_list.append(solpos.elevation[i])
            az_list.append(solpos.azimuth[i])
            dist_list.append(dist)

            distances.append(dist)
            cat = int(dist)
            while (cat + dist_width) < dist:
                cat += dist_width
            
            if cat in frequencies:
                frequencies[cat] += 1
            else:
                frequencies[cat] = 1

        date = date + datetime.timedelta(days=10)

    mu = torch.mean(torch.Tensor(distances))
    sig = torch.std(torch.Tensor(distances))
    x_range = torch.linspace(torch.min(torch.Tensor(distances)), torch.max(torch.Tensor(distances)), 100)
    ax.fill_between(x_range, nd(x_range, mu, sig), color=color, alpha=0.2)
    ax.plot(x_range, nd(x_range, mu, sig), color=color, alpha=alpha)

    sizes = [(size * d) ** 2 for d in distances]
    ax2.scatter(az_list, el_list, s=sizes, label=label, color=color, facecolors='none', edgecolors=color, alpha=alpha)

    # mean, var, skew, kurt = skewnorm.stats(4, moments='mvsk')
    # params = skewnorm.fit(torch.Tensor(distances))
    # ax.fill_between(x_range, skewnorm.pdf(x_range, *params), color=color, alpha=alpha)

    # ax.scatter([-1], [-1], label=label, color=color, marker='.', s=40 ** 2, alpha=0.6)
    
    return torch.max(torch.Tensor(distances)).item(), torch.max(nd(x_range, mu, sig)).item()

def plotDatasetSize(ax, data_points, label, color, alpha=0.4):
    label_dict = {'Training' : 1, 'Validation' : 2, 'Testing' : 3}
    ax.bar([label_dict[label]], [len(data_points)], label=label, color=color, alpha=alpha)

def main(dataset_path: str, dataset_config_path: str):

    with open(dataset_config_path, 'r') as file:
            dataset_config = json.load(file)

    dataset : HeliostatDataset = HeliostatDataset(data_points=dataset_path, dataset_config=dataset_config)

    # plot setup
    mpl.rcParams.update({'font.size': 24}) #32
    fig = plt.figure(figsize=(35.4,35.4 / 2), dpi=1000)
    # gs = mpl.gridspec.GridSpec(3,2, width_ratios=[3,1], height_ratios=[5, 5, 2], hspace=0.3, wspace=0.2)
    gs = mpl.gridspec.GridSpec(3,2, width_ratios=[1,1], hspace=0.3, wspace=0.2)
    time_ax = fig.add_subplot(gs[0])
    err_ax = fig.add_subplot(gs[1])
    angle_ax = fig.add_subplot(gs[2])
    dist_ax = fig.add_subplot(gs[4])
    gauss_ax = fig.add_subplot(gs[5])
    toi_ax = fig.add_subplot(gs[3])

    all_created_at = [dp.created_at for dp in dataset._data_points.values()]
    min_created_at = min(all_created_at)
    max_created_at = max(all_created_at)

    train_min_azim, train_max_azim, train_min_elev, train_max_elev = plotDataPointsOverAngles(ax=angle_ax, data_points=dataset.trainingDataset(), min_created_at=min_created_at, max_created_at=max_created_at, label='Training', color='orange', marker='*', size=150, dataset=dataset)
    valid_min_azim, valid_max_azim, valid_min_elev, valid_max_elev = plotDataPointsOverAngles(ax=angle_ax, data_points=dataset.testingDataset(), min_created_at=min_created_at, max_created_at=max_created_at, label='Validation', color='red', marker='D', size=100, dataset=dataset)
    test_min_azim, test_max_azim, test_min_elev, test_max_elev = plotDataPointsOverAngles(ax=angle_ax, data_points=dataset.evaluationDataset(), min_created_at=min_created_at, max_created_at=max_created_at, label='Testing', color=[51.0/255.0, 153.0/255, 1.0], marker='.', size=100, dataset=dataset)

    min_azim = min([train_min_azim, valid_min_azim, test_min_azim])
    max_azim = max([train_max_azim, valid_max_azim, test_max_azim])
    # min_azim = min([train_min_azim, test_min_azim])
    # max_azim = max([train_max_azim, test_max_azim])

    # min_elev = min([train_min_elev, valid_min_elev, test_min_elev])
    max_elev = max([train_max_elev, valid_max_elev, test_max_elev])
    # max_elev = max([train_max_elev, test_max_elev])

    plotDataPointsOverTime(ax=time_ax, data_points=dataset.trainingDataset(), min_created_at=min_created_at, max_created_at=max_created_at, label='Training', color='orange', marker='*', size=150)
    plotDataPointsOverTime(ax=time_ax, data_points=dataset.testingDataset(), min_created_at=min_created_at, max_created_at=max_created_at, label='Validation', color='red', marker='D', size=100)
    plotDataPointsOverTime(ax=time_ax, data_points=dataset.evaluationDataset(), min_created_at=min_created_at, max_created_at=max_created_at, label='Testing', color=[51.0/255.0, 153.0/255, 1.0], marker='.', size=100)

    plotErrFrequencies_1(err_ax, data_points=dataset.trainingDataset(), label='Training', color='orange', alpha=0.6, err_width=0.2)
    plotErrFrequencies_1(err_ax, data_points=dataset.testingDataset(), label='Validation', color='red', alpha=0.6, err_width=0.2)
    plotErrFrequencies_1(err_ax, data_points=dataset.evaluationDataset(), label='Testing', color='orange', alpha=0.6, err_width=0.2)

    # plotErrFrequencies_2(err_ax, data_points=dataset.trainingDataset(), label='Training', color=[51.0/255.0, 153.0/255, 1.0], alpha=0.6, err_width=0.2)
    # plotErrFrequencies_2(err_ax, data_points=dataset.testingDataset(), label='Validation', color='green', alpha=0.6, err_width=0.2)
    # plotErrFrequencies_2(err_ax, data_points=dataset.evaluationDataset(), label='Testing', color='orange', alpha=0.6, err_width=0.2)

    
    train_max_acc, train_max_d, dist_im = plotErrOverdist(dist_ax, data_points=dataset.trainingDataset(), min_created_at=min_created_at, max_created_at=max_created_at, label='Training', color='orange')
    valid_max_acc, valid_max_d, dist_im = plotErrOverdist(dist_ax, data_points=dataset.testingDataset(), min_created_at=min_created_at, max_created_at=max_created_at, label='Validation', color='red')
    test_max_acc, test_max_d, dist_im = plotErrOverdist(dist_ax, data_points=dataset.evaluationDataset(), min_created_at=min_created_at, max_created_at=max_created_at, label='Testing', color=[51.0/255.0, 153.0/255, 1.0])

    max_acc = max([train_max_acc, valid_max_acc, test_max_acc])
    max_d = max([train_max_d, valid_max_d, test_max_d])

    # max_acc = max([train_max_acc, test_max_acc])
    # max_d = max([train_max_d, test_max_d])

    date_range_str = dataset_config['all']['date_range']
    date_range = [datetime.datetime.strptime(date_range_str[0], "%Y-%m-%d %H:%M:%S"),datetime.datetime.strptime(date_range_str[1], "%Y-%m-%d %H:%M:%S")]
    train_max_dist, train_max_freq = plotGauss(gauss_ax, toi_ax, data_points=dataset.trainingDataset(), label='Training', color='orange', date_range=date_range)
    valid_max_dist, valid_max_freq = plotGauss(gauss_ax, toi_ax,  data_points=dataset.testingDataset(), label='Validation', color='red', date_range=date_range)
    test_max_dist, test_max_freq = plotGauss(gauss_ax, toi_ax, data_points=dataset.evaluationDataset(), label='Testing', color=[51.0/255.0, 153.0/255, 1.0], date_range=date_range)

    # plotDatasetSize(size_ax, data_points=dataset.trainingDataset(), label='Training', color=[51.0/255.0, 153.0/255, 1.0])
    # plotDatasetSize(size_ax, data_points=dataset.testingDataset(), label='Validation', color='green')
    # plotDatasetSize(size_ax, data_points=dataset.evaluationDataset(), label='Testing', color='orange')

    # max_dist = max([train_max_dist, test_max_dist, valid_max_dist])
    # max_freq = max([train_max_freq, test_max_freq, valid_max_freq])

    max_dist = max([train_max_dist, test_max_dist])
    max_freq = max([train_max_freq, test_max_freq])

    # formatting
    angle_ax.set_ylabel('Source Elevation [Deg]')
    angle_ax.set_xlabel('Source Azimuth [Deg]')
    angle_ax.set_xlim(0.95*min_azim, 1.05*max_azim)
    angle_ax.set_ylim(0, 1.05*max_elev)
    # angle_ax.legend()

    time_ax.set_ylabel('Hour')
    time_ax.set_xlabel('Date')
    time_ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
    time_ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=90))
    # time_ax.legend()

    err_ax.set_ylabel('Frequency')
    err_ax.set_xlabel('Accuracy Category ≤ [mRad]')
    err_ax.set_ylim(bottom=0)
    err_ax.set_xlim((0, max_acc*1.1))

    dist_ax.set_xlabel('Accuracy [mRad]')
    dist_ax.set_ylabel('Distance To Training [Rad]')
    dist_ax.set_ylim(bottom=0, top=max_d*1.1)
    dist_ax.set_xlim((0, max_acc*1.1))
    # dist_ax.colorbar('coolwarm')
    dist_ax.legend()

    divider = make_axes_locatable(dist_ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    # dist_im = dist_ax.scatter(-1,-1,alpha=0)
    cmap = plt.get_cmap('coolwarm')
    cbar = fig.colorbar(dist_im, cax=cax, orientation='vertical', cmap=cmap, ticks=[0.19, 2.35])
    cbar.ax.set_yticklabels(['Earliest', 'Latest'])


    gauss_ax.set_xlabel('Distance To Time Of Interest\n[Rad]')
    gauss_ax.set_ylabel('Frequency [%]\n')
    gauss_ax.set_xlim((0, max_dist * 1.1))
    gauss_ax.set_ylim(bottom=0)
    gauss_ax.set_yticks([])
    # gauss_ax.legend()

    toi_ax.set_xlim(0.95*min_azim, 1.05*max_azim)
    toi_ax.set_ylim(0, 1.05*max_elev)
    toi_ax.set_ylabel('Source Elevation [Deg]')
    toi_ax.set_xlabel('Source Azimuth [Deg]')

    # size_ax.set_xticks([])
    # size_ax.set_ylabel('Dataset Size')
    gauss_ax.scatter([-1], [-1], label='Training', color=[51.0/255.0, 153.0/255, 1.0], marker='.', s=40 ** 2, alpha=0.6)
    gauss_ax.scatter([-1], [-1], label='Validation', color='red', marker='.', s=40 ** 2, alpha=0.6)
    gauss_ax.scatter([-1], [-1], label='Testing', color='orange', marker='.', s=40 ** 2, alpha=0.6)
    # size_ax.set_xlim((0,1))
    # size_ax.set_ylim((0,1))
    # size_ax.set_xticks([])
    # size_ax.set_yticks([])
    # size_ax.axis('off')
    gauss_ax.legend()

    plt.savefig(os.path.abspath(os.path.join('/Users/moritz/Desktop', str(datetime.datetime.now())+ '.pdf')), bbox_inches='tight')

if __name__ == '__main__':
    data_dir = '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Gauss AM43 GM 3/18_02_2023_14_19/Results/SIMPLE'
    main(dataset_path=os.path.abspath(os.path.join(data_dir, 'dataset_data.csv')),
         dataset_config_path=os.path.abspath(os.path.join(data_dir, 'dataset_config.json')),
         )