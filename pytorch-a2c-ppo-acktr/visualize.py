# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})
from collections import deque

def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]


color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


def visdom_plot(viz, win, folder, game, name, num_steps, bin_size=100, smooth=1):
    tx, ty = load_data(folder, smooth, bin_size)
    if tx is None or ty is None:
        return win

    fig = plt.figure()
    plt.plot(tx, ty, label="{}".format(name))

    tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    ticks = tick_fractions * num_steps
    tick_names = ["{:.0e}".format(tick) for tick in ticks]
    plt.xticks(ticks, tick_names)
    plt.xlim(0, num_steps * 1.01)

    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    plt.title(game)
    plt.legend(loc=4)
    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    return viz.image(image, win=win)


def plot_line(viz, plot_window, opts_dict, data_point, count):
    if plot_window is None:
        plot_window = viz.line(X=np.array([count]),
                               Y=np.array([data_point]),
                               opts=opts_dict
                               )
    else:
        viz.line(X=np.array([count]),
                 Y=np.array([data_point]),
                 opts=opts_dict,
                 win=plot_window,
                 update='append')
    return plot_window

def plot_multi_lines(viz, plot_window, opts_dict, data_point, count):
    if plot_window is None:
        plot_window = viz.line(X=np.array([count]),
                               Y=data_point,
                               opts=opts_dict)
    else:
        viz.line(X=np.array([count]),
                 Y=data_point,
                 opts=opts_dict,
                 win=plot_window,
                 update='append')
    return plot_window

class VisdomPlotterA2C():
    def __init__(self, viz, plot_distill_loss = False):
        #from visdom import Visdom
        self.viz = viz
        #self.win = None
        self.plot_distill_loss = plot_distill_loss
        self.dist_plot_win, self.reward_plot_win, self.loss_plot_win = (None for _ in range(3))
        self.dist_plot_legend, self.reward_plot_legend, self.loss_plot_legend = self.get_legends(self.plot_distill_loss)
        self.dist_entropy_history = deque(maxlen=250)
        self.loss_history = deque(maxlen=250)
        self.reward_history = deque(maxlen=2500)

    def get_legends(self, plot_distill_loss):
        dist_entropy_opts = dict(
            xlabel="Frames in Mio.",
            ylabel="Entropy",
            title="Distribution Entropy"
        )
        reward_opts = dict(
            xlabel="Frames in Mio.",
            ylabel="Reward",
            title="Rewards",
            legend=["Mean Reward", "Median Reward"]
        )
        if plot_distill_loss:
            loss_legend = ["Value Loss", "Policy Loss", "Distill Loss"]
        else:
            loss_legend = ["Value Loss", "Policy Loss"]
        loss_opts = dict(
            xlabel="Frames in Mio.",
            ylabel="Loss",
            title="Loss",
            legend=loss_legend
        )
        return dist_entropy_opts, reward_opts, loss_opts

    def append(self, dist_entropy, reward, value_loss, action_loss, distill_loss=None):
        self.dist_entropy_history.append(dist_entropy)
        if self.plot_distill_loss:
            self.loss_history.append((value_loss, action_loss, distill_loss))
        else:
            self.loss_history.append((value_loss, action_loss))
        self.reward_history.extend(reward)

    def plot(self, frames):
        frames_in_mio = frames / 1000000
        self.dist_plot_win = plot_line(self.viz,
                                       self.dist_plot_win,
                                       self.dist_plot_legend,
                                       np.array(self.dist_entropy_history).mean(),
                                       frames_in_mio)
        loss_mean = np.mean(np.array(self.loss_history), axis=0)
        self.loss_plot_win = plot_line(self.viz,
                                       self.loss_plot_win,
                                       self.loss_plot_legend,
                                       loss_mean,
                                       frames_in_mio)
        if len(self.reward_history) >= self.reward_history.maxlen - 1:
            reward_mean_smooth = np.mean(self.reward_history)
            reward_median_smooth = np.median(self.reward_history)
            self.reward_plot_win = plot_multi_lines(self.viz,
                                                    self.reward_plot_win,
                                                    self.reward_plot_legend,
                                                    np.array([[reward_mean_smooth, reward_median_smooth]]),
                                                    frames_in_mio)



if __name__ == "__main__":
    from visdom import Visdom
    viz = Visdom()
    visdom_plot(viz, None, '/tmp/gym/', 'BreakOut', 'a2c', bin_size=100, smooth=1)
