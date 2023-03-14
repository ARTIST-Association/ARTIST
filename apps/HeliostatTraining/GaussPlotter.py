import json
import os
import typing

import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def getDataFromDir(dir : str) -> typing.Optional[float]:
    target_dir = os.path.join(dir, 'Results', 'SIMPLE')
    data_file = os.path.join(target_dir, 'training_results.json')
    if os.path.exists(target_dir):
        with open(data_file) as json_file:
            data = json.load(json_file)
            return data['Eval Deviation'] * 1000 #if data['Max Epoch'] >= 500 else None #and data['Eval Deviation'] <= 3) else None

    return None

def nd(x, mu, sig):
    # return (1 / np.sqrt(2 * np.pi * (sig**2))) * np.exp((-1/2) * ((x-mu) / sig)**2)
    return (1 / np.sqrt(2 * np.pi * (sig**2))) * np.exp((-1/2) * (x-mu)**2 / (sig**2))

def main(random_data_dir : str,
         max_hd_dirs : typing.Dict[int, str],
         time_method_dir : typing.Optional[str] = None):

    random_results = []
    for subdir, dirs, files in os.walk(random_data_dir):
        for dir in dirs:
            n_d = os.path.join(random_data_dir,dir)
            if os.path.exists(n_d):
                re = getDataFromDir(n_d)
                if re:
                    random_results.append(re)

    abs_v = np.abs(np.array(random_results) - np.mean(random_results))
    std_v = (2 * np.std(random_results))#
    index_v = abs_v < std_v
    random_results = np.array(random_results)[index_v]

    categories = {}
    expected_mean = np.mean(random_results)
    for i in range(0,400):
        categories[expected_mean + i * 0.005] = 0
        categories[expected_mean - i * 0.005] = 0

    max_num_in_cat = 0
    for rr in random_results:
        min_dist = None
        min_dist_cat = None
        for key in categories.keys():
            dist = abs(key - rr)
            if not min_dist or dist < min_dist:
                min_dist = dist
                min_dist_cat = key
        
        categories[min_dist_cat] = categories[min_dist_cat] + 1
        if categories[min_dist_cat] > max_num_in_cat:
            max_num_in_cat = categories[min_dist_cat]

    categories_accumulated = {}
    for k in categories.keys():
        num_accumulated = 0
        for k2, v in categories.items():
            if k2 >= k:
                num_accumulated += v

        categories_accumulated[k] = num_accumulated

    max_results = {}
    for num_nn, max_hd_dir in max_hd_dirs.items():
        max_results[num_nn] = []
        for subjdir, dirs, files in os.walk(max_hd_dir):
            for dir in dirs:
                n_d = os.path.join(max_hd_dir,dir)
                if os.path.exists(n_d):
                    re = getDataFromDir(n_d)
                    if re:
                        max_results[num_nn].append(re)

        abs_v = np.abs(np.array(max_results[num_nn]) - np.mean(max_results[num_nn]))
        std_v = (2 * np.std(max_results[num_nn]))
        if std_v > 0.0:
            index_v = abs_v < std_v
            max_results[num_nn] = np.array(max_results[num_nn])[index_v]
        else:
            max_results[num_nn] = np.array(max_results[num_nn])
    
    max_values = {}
    for num_nn in max_results.keys():
        max_mean = np.mean([m for m in max_results[num_nn] if m <= 10])
        max_std = np.std([m for m in max_results[num_nn] if m <= 10])
        max_values[num_nn] = (max_mean, max_std)

    if time_method_dir:
        time_results = []
        for subdir, dirs, files in os.walk(time_method_dir):
            for dir in dirs:
                n_d = os.path.join(time_method_dir,dir)
                if os.path.exists(n_d):
                    re = getDataFromDir(n_d)
                    if re:
                        time_results.append(re)
                        
        time_value = 0
        for mr in time_results:
            time_value += mr / len(time_results)

    fig = plt.figure(figsize=(35.4/2,10), dpi=1000)
    plt_ax1 = fig.add_subplot(2,1,1)
    plt_ax2 = fig.add_subplot(2,1,2)

    c_dict = {1 : 'orange', 2 : 'red', 3 : 'cyan', 4: 'green', 5: 'blue'}
    
    y_bound = max_num_in_cat * 1.05 / len(random_results) * 100

    x_mean = np.mean([r for r in random_results if r < 2])
    x_std = np.std([r for r in random_results if r < 2])
    # x_range = np.linspace(x_mean - x_std * 4, x_mean + x_std * 4, 100)
    x_range = np.linspace(min(random_results), max(random_results), 100)
    plt_ax1.fill_between(x_range, nd(x_range, x_mean, x_std), color='grey', alpha = 0.3, label='Random')
    plt_ax1.bar([k for k,v in categories.items() if v > 0], [float(v) / float(len(random_results)) * 100.0 for k,v in categories.items() if v > 0], width = 0.005, color='grey', alpha = 0.1)
    for num_nn, max_value in max_values.items():
        x_range = np.linspace(min(max_results[num_nn]), max(max_results[num_nn]), 100)
        # plt_ax1.fill_between(x_range, nd(x_range, max_value[0], max_value[1]), color=c_dict[num_nn], alpha=0.2, label=str(num_nn)+'-NN (' + str(len(max_results[num_nn])) + ')')
        plt_ax1.plot([max_value[0], max_value[0]], [0, 110], c=c_dict[num_nn], linewidth=7, linestyle='dashed', alpha=0.6, label=str(num_nn)+'-NN')
    if time_method_dir:
        plt_ax1.plot([time_value, time_value], [0, y_bound], c='magenta', linewidth=7, label='Date', alpha=0.6)
    # plt_ax1.set_xlabel('Evaluation Performance [mRad]')
    # plt_ax1.set_xlim((min(random_results) - 0.025, max(random_results) + 0.075))
    plt_ax1.set_xlim((0.55, 1.6 * max([k for k,v in categories.items() if v > 0])))
    plt_ax1.set_ylim((0,1.05 * y_bound))
    plt_ax1.set_ylabel('Frequency\n[%]')

    plt_ax2.bar([k for k,v in categories_accumulated.items() if v > 0], [v / len(random_results) * 100 for k,v in categories_accumulated.items() if v > 0], width = 0.005, color='grey', label='random', alpha = 0.3)
    for num_nn, max_value in max_values.items():
        plt_ax2.plot([max_value[0], max_value[0]], [0, 110], c=c_dict[num_nn], linewidth=7, linestyle='dashed', alpha=0.6, label=str(num_nn)+'-NN')
    if time_method_dir:
        plt_ax2.plot([time_value, time_value], [0, 110], c='magenta', linewidth=7, label='Date', alpha=0.6,)
    plt_ax2.set_xlabel('Testing Performance [mRad]')
    # plt_ax2.set_xlim((min(random_results) - 0.025, max(random_results) + 0.0725))
    plt_ax2.set_xlim(plt_ax1.get_xlim())
    plt_ax2.set_ylim((0,110))
    plt_ax2.set_ylabel('Accumulated\nFrequency [%]')
    plt_ax1.set_xticks([])

    fig.suptitle('Total Number Of Randomized Trainings: ' + str(len(random_results)))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    plt_ax1.legend(prop={'size': 24})
    # plt_ax2.legend()
    plt.savefig(os.path.abspath(os.path.join('/Users/moritz/Desktop', str(datetime.datetime.now())+ '.pdf')), bbox_inches='tight')

if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 24})
    main(random_data_dir= '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Gauss AJ23 GM Random',
         max_hd_dirs = {
            1: '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Gauss AJ23 GM 1',
            2: '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Gauss AJ23 GM 2',
            3: '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Gauss AJ23 GM 3',
            4: '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Gauss AJ23 GM 4',
            5: '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Gauss AJ23 GM 5',
            },
         time_method_dir ='/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Gauss AJ23 Date Sweep GM',
            )