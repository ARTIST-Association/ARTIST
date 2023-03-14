import matplotlib.pyplot as plt
import matplotlib
import typing
import os
import json
import datetime

def getDataFromDir(dir : str) -> typing.Optional[float]:
    
    data_file = os.path.join(dir, 'dataset_config.json')
    results_dir = os.path.join(dir, 'Results', 'SIMPLE')
    results_file = os.path.join(results_dir, 'training_results.json')

    dt = None
    l = None
    with open(data_file) as json_file:
        data = json.load(json_file)
        dt = data['testing']['date_range'][0]
        l_train = len(data['training']['data points'])
        l_test = len(data['testing']['data points'])
        l_eval = len(data['evaluation']['data points'])
        l = [l_train, l_test, l_eval]

    re = None
    
    if os.path.exists(results_dir):
        with open(results_file) as json_file:
            data = json.load(json_file)
            re_train = data['Train Deviation'] * 1000
            re_test = data['Test Deviation'] * 1000
            re_eval = data['Eval Deviation'] * 1000

    else:
        return [None, None, None]
    
    return [re_train, re_test, re_eval], dt, l

def main(date_dir : str):
    
    train_results = {}
    test_results = {}
    eval_results = {}
    train_amount = {}
    test_amount = {}
    eval_amount = {}
    for subdir, dirs, files in os.walk(date_dir):
        for dir in dirs:
            n_d = os.path.join(date_dir,dir)
            if os.path.exists(n_d):
                re, dt, l = getDataFromDir(n_d)
                if re and dt and re[2] < 100:
                    dt = datetime.datetime.strptime(dt,"%Y-%m-%d %H:%M:%S")
                    train_results[dt] = re[0]
                    test_results[dt] = re[1]
                    eval_results[dt] = re[2]
                    train_amount[dt] = l[0]
                    test_amount[dt] = l[1]
                    eval_amount[dt] = l[2]

    fig = plt.figure(figsize=(30,10))
    plt_ax1 = fig.add_subplot(4,1,1)
    plt_ax1.plot([key for key in sorted(train_results)], [train_results[key] for key in sorted(train_results)], c='blue')
    plt_ax1.scatter([key for key in sorted(train_results) if train_results[key] <= 1.0], [train_results[key] for key in sorted(train_results) if train_results[key] <= 1.0], c='blue', marker='*', s=80)
    # plt_ax1.plot([key for key in sorted(train_results)], [1 for key in sorted(train_results)], c='grey')
    plt_ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt_ax1.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=30))

    plt_ax2 = fig.add_subplot(4,1,2)
    plt_ax2.plot([key for key in sorted(test_results)], [test_results[key] for key in sorted(test_results)], c='green')
    plt_ax2.scatter([key for key in sorted(test_results) if test_results[key] <= 1.0], [test_results[key] for key in sorted(test_results) if test_results[key] <= 1.0], c='green', marker='*', s=80)
    # plt_ax2.plot([key for key in sorted(test_results)], [1 for key in sorted(test_results)], c='grey')
    
    plt_ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt_ax2.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=30))

    plt_ax3 = fig.add_subplot(4,1,3)
    plt_ax3.plot([key for key in sorted(eval_results)], [eval_results[key] for key in sorted(eval_results)], c='orange')
    plt_ax3.scatter([key for key in sorted(eval_results) if eval_results[key] <= 1.0], [eval_results[key] for key in sorted(eval_results) if eval_results[key] <= 1.0], c='orange', marker='*', s=80)
    # plt_ax3.plot([key for key in sorted(eval_results)], [1 for key in sorted(eval_results)], c='grey')
    
    plt_ax3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt_ax3.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=30))

    plt_ax4 = fig.add_subplot(4,1,4)
    plt_ax4.plot([key for key in sorted(train_results)], [train_amount[key] for key in sorted(train_results)], c='blue')
    plt_ax4.plot([key for key in sorted(test_results)], [test_amount[key] for key in sorted(test_results)], c='green')
    plt_ax4.plot([key for key in sorted(eval_results)], [eval_amount[key] for key in sorted(eval_results)], c='orange')
    plt_ax4.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt_ax4.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=30))

    fig.autofmt_xdate()

    plt.show()

if __name__ == '__main__':
    main('/Users/Synhelion/Desktop/MA/UltraCalibMacOS/Synhelion Codebase/Python/apps/HeliostatTraining/TrainingBase/Dates AM43')