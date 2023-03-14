import typing
import os
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch

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

def get_avg_data(data_dir : str):
    results = []
    index = 2
    for subdir, dirs, files in os.walk(data_dir):
        for dir in dirs:
            n_d = os.path.join(data_dir,dir)
            if os.path.exists(n_d):
                re = getDataFromDir(n_d)
                if re:
                    results.append(re[index])

    return np.mean(results), len(results)

def main(ref_dir : str, data_dirs : typing.List[str], labels : typing.List[int]):
    matplotlib.rcParams.update({'font.size': 22})
    
    ref_result, ref_num = get_avg_data(data_dir=ref_dir)

    regu_results = {}
    regu_nums = {}
    for dir, label in zip(data_dirs, labels):
        regu_results[label], regu_nums[label] = get_avg_data(data_dir=dir)
    regu_results = dict(sorted(regu_results.items()))
    regu_nums = dict(sorted(regu_nums.items()))
    regu_facs = [regu_fac for regu_fac in regu_results.keys()]
    regu_vals = [re for re in regu_results.values()]
    regu_nums = [n for n in regu_nums.values()]

    fig = plt.figure(figsize=(30,10))
    plt_ax1 = fig.add_subplot(1,1,1)
    plt_ax1_2 = plt_ax1.twinx()


    plt_ax1.plot([0, regu_facs[-1]], [ref_result, ref_result], c='magenta', label='static (Rigid Body)')
    # plt_ax1.plot([0, regu_facs[-1]], [regu_results[0], regu_results[0]], c='orange', alpha=0.2)
    plt_ax1.plot(regu_facs[1:], regu_vals[1:], c='orange', label='Pre-Aligned NN')
    plt_ax1.scatter(regu_facs[1:], regu_vals[1:], c='orange')

    plt_ax1_2.plot(regu_facs[1:], regu_nums[1:], c='blue')

    plt_ax1.set_xscale('log')
    plt_ax1.set_xlim((0,regu_facs[-1]))

    plt_ax1.set_xlabel('Weight Decay Factor')
    plt_ax1.set_ylabel('Mean Validation Perfomance [mRad]')
    plt_ax1.legend()
    plt.show()

if __name__ == '__main__':
    ref_dir = '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Regu Sweep AM35 GM'
    data_dirs = [
        '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Regu Sweep AM35 PA',
        '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Regu Sweep AM35 PA 1000',
        '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Regu Sweep AM35 PA 100',
        '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Regu Sweep AM35 PA 10',
        '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Regu Sweep AM35 PA 1',
        '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Regu Sweep AM35 PA 0.1',
        '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Regu Sweep AM35 PA 0.01',
        '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Regu Sweep AM35 PA 0.001',
        # '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Regu Sweep AM35 PA 0.0001',
        # '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Regu Sweep AM35 PA 0.00001',
        # '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Regu Sweep AM35 PA 0.000001',
        # '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Regu Sweep AM35 PA 0.0000001',
        # '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Regu Sweep AM35 PA 0.00000001',
        
        
        
    ]
    labels = [
        0.0,
        1000,
        100,
        10,
        1,
        0.1,
        0.01,
        0.001,
        # 0.0001,
        # 0.00001,
        # 0.000001,
        # 0.0000001,
        # 0.00000001,
    ]
    main(ref_dir=ref_dir, data_dirs=data_dirs, labels=labels)