import numpy as np
from collections import deque


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

class VisdomPlotGraph():

    def __init__(self, viz, title, ylabel, legend = None,
                 running_mean_n = 250, plot_after_n_inserts = 50):
        self.viz = viz
        # self.win = None
        self.plot_win = None

        if legend:
            self.plot_legend = dict(
                xlabel="Frames in Mio.",
                ylabel=ylabel,
                title=title,
                legend=legend
            )
        else:
            self.plot_legend = dict(
                xlabel="Frames in Mio.",
                ylabel=ylabel,
                title=title
            )
        self.history = deque(maxlen=running_mean_n)
        self.plot_after_history_size = min(running_mean_n, plot_after_n_inserts)


    def append(self, value):
        self.history.append(value)

    def extend(self, value):
        self.history.extend(value)

    def plot_values(self, frames, values):
        if self.viz is not None and len(self.history) >= self.plot_after_history_size:
            frames_in_mio = frames / 1000000
            self.plot_win = plot_line(self.viz,
                                      self.plot_win,
                                      self.plot_legend,
                                      values,
                                      frames_in_mio)

    def plot(self, frames):
        values = np.mean(np.array(self.history), axis=0)
        self.plot_values(frames, values)

    def plot_median_and_mean(self, frames):
        mean_smooth = np.mean(self.history)
        median_smooth = np.median(self.history)
        values = np.array([mean_smooth, median_smooth])
        self.plot_values(frames, values)

    def get_mean(self):
        return np.mean(np.array(self.history), axis=0)

    def get_median(self):
        return np.median(self.history)


class VisdomPlotterA2C():
    def __init__(self, viz, plot_distill_loss = False,
                 entropy_plot_cnf = [2000, 200], reward_plot_cnf=[20000, 2000], loss_plot_cnf = [2000, 200]):
        # from visdom import Visdom
        self.viz = viz
        self.plot_distill_loss = plot_distill_loss
        self.dist_entropy_plotter = VisdomPlotGraph(viz,
                                                    "Distribution Entropy",
                                                    "Entropy",
                                                    running_mean_n=entropy_plot_cnf[0],
                                                    plot_after_n_inserts=entropy_plot_cnf[1])
        self.reward_plotter = VisdomPlotGraph(viz,
                                              "Reward",
                                              "Reward",
                                              ["Mean Reward", "Median Reward"],
                                              running_mean_n=reward_plot_cnf[0],
                                              plot_after_n_inserts=reward_plot_cnf[1])

        if self.plot_distill_loss:
            loss_legend = ["Value Loss", "Policy Loss", "Distill Loss"]
        else:
            loss_legend = ["Value Loss", "Policy Loss"]
        self.loss_plotter = VisdomPlotGraph(viz, "Loss", "Loss", loss_legend,
                                            running_mean_n = loss_plot_cnf[0],
                                            plot_after_n_inserts=loss_plot_cnf[1])


    def append(self, dist_entropy, reward, value_loss, action_loss, distill_loss=None):
        self.dist_entropy_plotter.append(dist_entropy)
        self.reward_plotter.extend(reward)
        if self.plot_distill_loss:
            self.loss_plotter.append((value_loss, action_loss, distill_loss))
        else:
            self.loss_plotter.append((value_loss, action_loss))

    def plot(self, frames):
        self.dist_entropy_plotter.plot(frames)
        self.reward_plotter.plot_median_and_mean(frames)
        self.loss_plotter.plot(frames)


class VisdomPlotterEM():
    def __init__(self, viz):
        running_mean_n = 5000
        self.loss_plotter = VisdomPlotGraph(viz, "Loss", "Loss", ["State Loss", "Reward Loss"], running_mean_n, 200)
        self.reward_plotter = VisdomPlotGraph(viz, "Reward", "Reward", ["True Reward", "Predicted Reward"], 1, 0)

    def append(self, loss, reward):
        self.loss_plotter.append(loss)
        self.reward_plotter.append(reward)

    def plot(self, frames):
        self.loss_plotter.plot(frames)
        self.reward_plotter.plot(frames)

    def get_smoothed_values(self):
        loss = self.loss_plotter.get_mean()
        return loss[0], loss[1]
