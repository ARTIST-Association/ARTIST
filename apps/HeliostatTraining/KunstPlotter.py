import json
import os
import typing

import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def nd(x, mu, sig):
    # return (1 / np.sqrt(2 * np.pi * (sig**2))) * np.exp((-1/2) * ((x-mu) / sig)**2)
    return (1 / np.sqrt(2 * np.pi * (sig**2))) * np.exp((-1/2) * (x-mu)**2 / (sig**2))

def getDataFromDir(dir : str) -> typing.Optional[float]:
    target_dir = os.path.join(dir, 'Results', 'SIMPLE')
    data_file = os.path.join(target_dir, 'training_results.json')
    if os.path.exists(target_dir):
        with open(data_file) as json_file:
            data = json.load(json_file)
            # if data['Eval Deviation'] * 1000 > 1.7:
            #     print(target_dir)
            return [data['Eval Deviation'] * 1000, data['Evaluation Distance']] #if data['Max Epoch'] >= 500 else None #and data['Eval Deviation'] <= 3) else None
    return None

def getDataFromDirs(sup_dir : str) -> typing.List[float]:
    results = [[], []]
    for subdir, dirs, files in os.walk(sup_dir):
        for dir in dirs:
            n_d = os.path.join(sup_dir,dir)
            if os.path.exists(n_d):
                re = getDataFromDir(n_d)
                if re:
                    results[0].append(re[0])
                    results[1].append(re[1])

    return results


def plot_gauss(ax, data, label, color, index = 0):
    # err_range = np.linspace(min(data), max(data), 100)
    # err_range = np.linspace(np.mean(data) - np.std(data), np.mean(data) + np.std(data), 100)
    # ax.fill_between(err_range, nd(err_range, np.mean(data), np.std(data)) + 0.1 * index, 0.1 * index, color=color, alpha=0.3)
    # ax.fill_between(err_range, nd(err_range, np.mean(data), np.std(data)), 0.1 * index, color=color, alpha=0.3)
    # ax.plot([np.mean(data), np.mean(data)], [0, 1], color=color, label = label)
    # ax.scatter(data, np.ones(len(data)) * (0.1 * index + 1), color=color, alpha = 0.7, s=3)
    x = data[:][0]
    y = data[:][1]
    ax.scatter(y, x, color=color, alpha = 0.7, s=7, label=label)


def main(gauss_dirs, colors):
    matplotlib.rcParams.update({'font.size': 24})

    data_dir = {}
    for key, d in gauss_dirs.items():
        data_dir[key] = getDataFromDirs(sup_dir = d)

    fig = plt.figure(figsize=(35.4/2,10), dpi=1000)
    gauss_ax = fig.add_subplot(1,1,1)
    # plt_ax2 = fig.add_subplot(2,1,2)

    print('Plotting')
    for i, key in enumerate(data_dir.keys()):
        plot_gauss(gauss_ax, data_dir[key], label=key, color=colors[key], index=i)

    gauss_ax.legend()

    plt.savefig(os.path.abspath(os.path.join('/Users/moritz/Desktop', str(datetime.datetime.now())+ '.pdf')), bbox_inches='tight')
        
if __name__ == '__main__':
    gauss_dirs = {
        'random' : '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Gauss AJ23 Date Sweep GM Random',
        'dates' : '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Gauss AJ23 Date Sweep GM',
        '1-NN' : '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Gauss AJ23 Date Sweep GM NN1',
        '2-NN' : '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Gauss AJ23 Date Sweep GM NN2',
        '3-NN' : '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Gauss AJ23 Date Sweep GM NN3',
        '4-NN' : '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Gauss AJ23 Date Sweep GM NN4',
        '5-NN' : '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Gauss AJ23 Date Sweep GM NN5',
    }

    colors = {
        'random' : 'grey',
        'dates' : 'black',
        '1-NN' : 'cyan',
        '2-NN' : 'green',
        '3-NN' : 'orange',
        '4-NN' : 'magenta',
        '5-NN' : 'pink',
    }

    main(gauss_dirs=gauss_dirs, colors=colors)