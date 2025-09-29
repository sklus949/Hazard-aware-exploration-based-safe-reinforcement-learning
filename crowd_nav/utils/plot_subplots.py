
# encoding: utf-8
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.ticker as ticker

parser = argparse.ArgumentParser()
parser.add_argument('--window_size', type=int, default=500)
args = parser.parse_args()
rcParams['font.family'] = 'Times New Roman'


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def main():
    # define the names of the models you want to plot and the longest episodes you want to show
    max_episodes = 50000
    log_file = '../data/output/output.log'  # 日志文件
    with open(log_file, 'r') as file:
        log = file.read()

    # 匹配日志文件中的数据
    train_pattern = r"TRAIN in episode (.*) has success rate: (.*), collision rate: (.*), nav time: (.*), total reward: (.*)"
    # train_pattern = r"TRAIN in episode 19984 has success rate: 1.00, collision rate: 0.00, nav time: 8.50, total reward: 8.5119"

    train_reward = []
    infolist = []
    navtime = []
    success_rate = 0
    timeout_rate = 0
    collision_rate = 0
    for r in re.findall(train_pattern, log):
        train_reward.append(float(r[1]))
        infolist.append((r[4]))
        navtime.append(float(r[3]))

    for info in infolist[-500:]:
        if info == 'Reaching goal':
            success_rate += 1
        if info == 'Collision':
            collision_rate += 1
        if info == 'Timeout':
            timeout_rate += 1

    train_reward = train_reward[:max_episodes]
    # print train_reward
    train_reward_smooth = running_mean(train_reward, args.window_size)
    train_nav_time = running_mean(navtime, args.window_size)

    ax_legends = ['ours']

    # 设置刻度格式为K
    def format_ticks(x, pos):
        return f'{x / 1000:.0f}k'

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for x in ax:
        x.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
    ax[0].plot(range(len(train_reward_smooth)), train_reward_smooth, color="C3", linewidth=1, label='GT')
    ax[0].plot(range(len(train_reward_smooth)), train_reward_smooth, color="C3", linewidth=1, label='GT')
    ax[0].set_title("Title1")
    ax[0].legend(ax_legends, shadow=True, loc='best', prop={'size': 14, 'family': 'Times New Roman'})

    ax[1].plot(range(len(train_nav_time)), train_nav_time, color="C5", linewidth=1, label='GT')
    ax[1].set_title("Title2")
    ax[1].legend(ax_legends, shadow=True, loc='best', prop={'size': 14, 'family': 'Times New Roman'})

    ax = plt.gca()
    ax.patch.set_facecolor('xkcd:white')

    # ax.grid(True) # 设置网格
    # ax.spines['top'].set_visible(False)  # 去掉上边框
    # ax.spines['right'].set_visible(False)  # 去掉右边框
    # ax.patch.set_facecolor("green")
    ax.patch.set_alpha(0.5)

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.show()


if __name__ == '__main__':
    main()
