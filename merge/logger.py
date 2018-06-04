import os
from visdom_plotter import VisdomPlotterEM
import time
import numpy as np


class LogFile():
    def __init__(self, log_file, delete_log_file = True):
        self.log_file = log_file
        basedir = os.path.dirname(self.log_file)
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        if delete_log_file == True and os.path.exists(self.log_file):
            os.remove(self.log_file)
        if not os.path.exists(self.log_file):
            open(self.log_file, 'a').close()

    def log(self, message):
        with open(self.log_file, 'a') as file:
            file.write(message + '\n')


class LogTrainEM():
    def __init__(self, log_name, delete_log_file = True, viz = None):
        # this logger will both print to the console as well as the file
        if viz is not None:
            self.visdom_plotter = VisdomPlotterEM(viz=viz)
        self.logger = LogFile(os.path.join('logs', log_name), delete_log_file)

        self.frames = 0
        self.start_time = time.time()
        self.episode_time = time.time()

        self.state_loss_list = []
        self.reward_loss_list = []
        self.reward = 0
        self.reward_prediction = 0


    def log_loss_and_reward(self, loss, reward_prediction, reward):
        self.frames += 1

        state_loss, reward_loss = loss
        state_loss = state_loss.data.cpu().numpy()[0]
        reward_loss = reward_loss.data.cpu().numpy()[0]
        self.state_loss_list.append(state_loss)
        self.reward_loss_list.append(reward_loss)
        self.reward_prediction = reward_prediction.data.cpu().numpy()[0]
        self.reward = reward.data.cpu().numpy()[0]

        if self.visdom_plotter:
            self.visdom_plotter.append((state_loss, reward_loss), (self.reward, self.reward_prediction))

    def print_episode(self, episode):
        summary = self.get_step_summary(episode)

        print(summary)
        if self.visdom_plotter:
            self.visdom_plotter.plot(frames=self.frames)
        self.logger.log(summary)


    def get_step_summary(self, episode):

        iteration_suffix = 'mio'
        frames_in_mio = self.frames / 1000000

        duration = time.time() - self.episode_time
        self.episode_time = time.time()
        total_time = time.time() - self.start_time

        str_game = 'Episode: {0:.0f}\t'.format(episode)
        str_iteration = 'it: {0:.1f}{1}\t\t'.format(frames_in_mio, iteration_suffix)
        str_loss = 'frame loss: {0:.9f}\t reward loss: {1:.5f}\t\t'\
            .format(np.mean(self.state_loss_list), np.mean(self.reward_loss_list))
        str_reward = 'pred reward: {0:.3f}\t reward: {1:.3f}\t\t'\
            .format(self.reward_prediction, self.reward)
        str_time = 'duration: {0:.3f}s\t total duration: {1:.3f}min'.format(duration, total_time / 60.)
        return str_game + str_iteration + str_loss + str_reward + str_time
