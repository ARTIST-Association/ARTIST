import typing
import os
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import scipy.stats as stats

def computeRegression(data_x, data_y):
    data_x = torch.tensor(data_x, dtype=torch.float64)
    data_y = torch.tensor(data_y, dtype=torch.float64)

    # bx + a = y
    a = torch.tensor(1, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam(params=[a,b], lr=0.1)
    loss_criterion = torch.nn.MSELoss(size_average=False)
    for epoch in range(2000):
            optimizer.zero_grad()
            pred = b * data_x + a
            loss = loss_criterion(pred, data_y)
            # print(str(epoch) + ': ' + str(loss.detach().numpy()))
            loss.backward()
            optimizer.step()

    return b.item(),a.item()

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

def main(data_dir : str):
    matplotlib.rcParams.update({'font.size': 22})
    
    results = {}
    for subdir, dirs, files in os.walk(data_dir):
        for dir in dirs:
            n_d = os.path.join(data_dir,dir)
            if os.path.exists(n_d):
                re = getDataFromDir(n_d)
                if re:
                    if re in results:
                        num_re = results[re[0]][0] + 1
                        results[re[0]] = [num_re, results[re[0]][0] + np.array(re)]
                    else:
                        results[re[0]] = [1, np.array(re[1:])]

    for key in results.keys():
        results[key] = results[key][1] / results[key][0]

    results = dict(sorted(results.items(), key=lambda item: item[0]))

    fig = plt.figure(figsize=(30,10))
    plt_ax1 = fig.add_subplot(2,3,1)
    plt_ax2 = fig.add_subplot(2,3,2)
    plt_ax3 = fig.add_subplot(2,3,3)
    plt_ax4 = fig.add_subplot(2,3,4)
    plt_ax6 = fig.add_subplot(2,3,6)

    data_x = [k for k in results.keys()]
    data_y = [v[2] for v in results.values()]
    b1,a1 = computeRegression(data_x=[data_x[0], data_x[-1]], data_y=[data_y[0], data_y[-1]])
    stats.probplot(np.abs(np.array(data_y) - (b1 * np.array(data_x) + a1)), plot=plt_ax4)
    plt_ax4.set_title('')

    plt_ax1.plot([data_x[0], data_x[-1]], [b1*data_x[0]+a1,b1*data_x[-1]+a1], c='grey', alpha=0.2)
    plt_ax1.plot([k for k in results.keys()], [v[0] for v in results.values()], c='blue', label='train')
    plt_ax1.scatter([k for k in results.keys()], [v[0] for v in results.values()], c='blue')
    plt_ax1.plot([k for k in results.keys()], [v[1] for v in results.values()], c='green', label='test')
    plt_ax1.scatter([k for k in results.keys()], [v[1] for v in results.values()], c='green')
    plt_ax1.plot([k for k in results.keys()], [v[2] for v in results.values()], c='orange', label='eval')
    plt_ax1.scatter([k for k in results.keys()], [v[2] for v in results.values()], c='orange')
    plt_ax1.set_xlabel('Training Dataset Size')
    plt_ax1.set_ylabel('Prediction Accuracy [mRad]')
    plt_ax1.legend()

    # plt_ax2_2 = plt_ax2.twinx()
    plt_ax2.plot([k for k in results.keys()], [v[3] for v in results.values()], c='grey')
    plt_ax2.scatter([k for k in results.keys()], [v[3] for v in results.values()], c='grey')
    plt_ax2.set_xlabel('Training Dataset Size')
    plt_ax2.set_ylabel('Mean Evaluation Distance [Rad]')
    # plt_ax2_2.plot([k for k in results.keys()], [v[4] for v in results.values()], c='orange')
    # plt_ax2_2.scatter([k for k in results.keys()], [v[4] for v in results.values()], c='orange')
    # plt_ax2_2.set_ylabel('Num Evaluation Data')

    results = dict(sorted(results.items(), key=lambda item: item[1][3]))
    data_x = [v[3] for v in results.values()]
    # data_x = [data_x[0], data_x[-1]]
    data_y = [v[2] for v in results.values()]
    # data_y = [data_y[0], data_y[-1]]
    b2,a2 = computeRegression(data_x=[data_x[0], data_x[-1]], data_y=[data_y[0], data_y[-1]])
    plt_ax3.plot([data_x[0], data_x[-1]], [b2*data_x[0]+a2,b2*data_x[-1]+a2], c='grey', alpha=0.2)
    plt_ax3.plot([v[3] for v in results.values()], [v[0] for v in results.values()], c='blue', label='train')
    plt_ax3.scatter([v[3] for v in results.values()], [v[0] for v in results.values()], c='blue')
    plt_ax3.plot([v[3] for v in results.values()], [v[1] for v in results.values()], c='green', label='test')
    plt_ax3.scatter([v[3] for v in results.values()], [v[1] for v in results.values()], c='green')
    plt_ax3.plot([v[3] for v in results.values()], [v[2] for v in results.values()], c='orange', label='eval')
    plt_ax3.scatter([v[3] for v in results.values()], [v[2] for v in results.values()], c='orange')
    plt_ax3.set_xlabel('Mean Evaluation Distance [Rad]')
    plt_ax3.set_ylabel('Prediction Accuracy [mRad]')
    plt_ax3.legend()

    stats.probplot(np.abs(np.array(data_y) - (b2 * np.array(data_x) + a2)), plot=plt_ax6)
    plt_ax6.set_title('')
    plt_ax6.set_ylim((plt_ax6.get_ylim()[0],plt_ax4.get_ylim()[1]))
    plt_ax6.set_xlim(plt_ax4.get_xlim())
    plt_ax4.set_ylim(plt_ax6.get_ylim())
    plt.show()

if __name__ == '__main__':
    data_dir = '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results/Size Sweep AJ23 GM'
    main(data_dir=data_dir)