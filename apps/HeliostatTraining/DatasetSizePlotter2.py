import typing
import os
import sys

latex_path = '/Library/TeX/texbin/latex /Library/TeX/texbin/man/man1/latex.1'
sys.path.append(latex_path)

import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import datetime

# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['ps.fonttype'] = 42
# mpl.rcParams['font.family'] = 'Arial'

def getDataFromDir(dir : str) -> typing.Optional[float]:
    target_dir = os.path.join(dir, 'Results', 'SIMPLE')
    data_file = os.path.join(target_dir, 'training_results.json')
    if os.path.exists(target_dir):
        with open(data_file) as json_file:
            data = json.load(json_file)
            return data['Num Training'], \
                   data['Train Deviation'] * 1000, \
                   data['Test Deviation'] * 1000, \
                   data['Eval Deviation'] * 1000, \
                   data['Evaluation Distance'], \
                   data['Num Evaluation']
    return None

def addDataToPlot(data_dir : str, size_plot, label: str, color : str, dashes : str):
    results = {}
    for subdir, dirs, files in os.walk(data_dir):
        for dir in dirs:
            n_d = os.path.join(data_dir,dir)
            if os.path.exists(n_d):
                re = getDataFromDir(n_d)
                if re:
                    if re[0] in results:
                        num_re = results[re[0]][0] + 1
                        results[re[0]] = [num_re, results[re[0]][1] + np.array(re)[1:]]
                    else:
                        results[re[0]] = [1, np.array(re[1:])]

    for key in results.keys():
        results[key] = [results[key][0], results[key][1] / results[key][0]]

    results = dict(sorted(results.items(), key=lambda item: item[0]))
    index = 2
    size_plot.plot([k for k in results.keys()], [v[1][index] for v in results.values()], c=color, linestyle=dashes, linewidth=3, label=label)
    size_plot.scatter([k for k in results.keys()], [v[1][index] for v in results.values()], c=color)
    # dist_plot.plot([v[1][3] for v in results.values()], [v[1][index] for v in results.values()], c=color, label=label)
    # dist_plot.scatter([v[1][3] for v in results.values()], [v[1][index] for v in results.values()], c=color)

    return [k for k in results.keys()], [v[1][3] for v in results.values()], [v[0] for v in results.values()]

def main(data_dirs : typing.List[str], labels : typing.List[str], colors : typing.List[str], dash_list : typing.List[str]):
    mpl.rcParams.update({'font.size': 24})

    fig = plt.figure(figsize=(35.4/2,14), dpi=1000)
    
    plt_ax2 = fig.add_subplot(2,1,2)
    plt_ax1 = fig.add_subplot(2,1,1)
    
    # plt_ax3 = fig.add_subplot(1,3,3)
    # plt_ax2_2 = plt_ax2.twinx()

    for data_dir, label, color, dashes in zip(data_dirs, labels, colors, dash_list):
        size_array, dist_array, nt_array = addDataToPlot(data_dir=data_dir, size_plot=plt_ax1, label=label, color=color, dashes=dashes)
        # plt_ax2_2.plot(size_array, nt_array, color=color)
    
    plt_ax2.plot(size_array, dist_array, c='grey', linewidth=3)
    plt_ax2.scatter(size_array, dist_array, c='grey')
    plt_ax2.set_xlabel('Training Dataset Size')
    plt_ax2.set_ylabel('Testing Distance [Rad]')
    # plt_ax2_2.set_ylabel('Number Of Trainings')
    # plt_ax2.setp(plt_ax2)

    # plt_ax1.set_xlabel('Training Dataset Size')
    # plt_ax2.set_xticks([])
    plt_ax1.set_ylabel('Testing Perfomance [mRad]')
    plt_ax1.legend(frameon=False)
    plt_ax1.xaxis.set_ticks_position('top')

    plt_ax1.grid()
    plt_ax2.grid()

    # plt_ax3.set_xlabel('Mean Testing Distance [Rad]')
    # plt_ax3.set_ylabel('Mean Testing Perfomance [mRad]')
    # plt_ax3.legend()

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)

    plt.savefig(os.path.abspath(os.path.join('/Users/moritz/Desktop', str(datetime.datetime.now())+ '.pdf')), bbox_inches='tight')

if __name__ == '__main__':
    data_dirs = [
        
        # '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Size Sweep AJ23 GM 6',
        # '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Size Sweep AJ23 GM 14',
        '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Size Sweep AM43 GM',
        '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Size Sweep AM43 NN',
        '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Size Sweep AM43 PA',
    ]

    labels = [
        # 'Static 6',
        # 'Static 14',
        'Static 20',
        'Neural Network',
        'Pre-Aligned'
    ]

    colors = [
        
        # 'aquamarine',
        # 'red',
        'magenta',
        'cyan',
        'orange',
    ]

    dash_list = [
        
        # 'solid',
        # (0, (3, 5, 1, 5, 1, 5)),
        'dotted',
        'dashed',
        'dashdot',
    ]

    main(data_dirs=data_dirs, labels=labels, colors=colors, dash_list=dash_list)