import os
from rl_visualization.visdom_plotter import VisdomPlotterEM
from rl_visualization.logger import LogFile
import time
import sys


class LoggingMiniPacmanEnvTraining():
    def __init__(self, log_name, batch_size = 1, delete_log_file = True, viz = None):
        # this logger will both print to the console as well as the file
        self.visdom_plotter = VisdomPlotterEM(viz=viz)
        self.logger = LogFile(log_name, delete_log_file)

        self.logger.log('command line args: ' + " ".join(sys.argv) + '\n')

        self.frames = 0
        self.batch_size = batch_size
        self.start_time = time.time()

        self.print_interval = 10
        self.reward_prediction = 0.
        self.reward = 0.


    def log_loss_and_reward(self, loss, prediction, sample, episode):
        self.frames += self.batch_size

        state_loss, reward_loss = loss
        state_loss = state_loss.item()
        reward_loss = reward_loss.item()

        next_state_prediction, reward_prediction = prediction
        # TODO fix for bit array
        self.reward_prediction = reward_prediction.detach().cpu().view(-1).numpy()[0]

        (state, action, next_state, reward) = sample
        self.reward = reward.detach().cpu().view(-1).numpy()[0]

        self.visdom_plotter.append((state_loss, reward_loss),
                                   (self.reward, self.reward_prediction))

        if self.frames % (self.print_interval * self.batch_size) == 0:
            self.print_episode(episode=episode)

    def print_episode(self, episode):
        summary = self.get_step_summary(episode)

        print(summary)
        if self.visdom_plotter:
            self.visdom_plotter.plot(frames=self.frames)
        self.logger.log(summary)


    def get_step_summary(self, episode):
        total_time = time.time() - self.start_time

        state_loss, reward_loss = self.visdom_plotter.get_smoothed_values()

        loss_info = "frame loss {:.5f}, reward loss {:.5f}".format(state_loss, reward_loss)
        reward_info = "pred reward {:.5f}, reward {:.5f}".format(self.reward_prediction, self.reward)

        info = "Updates {}, num timesteps {}, FPS {}, {}, {}, time {:.5f} min" \
            .format(episode, self.frames, int(self.frames / (total_time)),
                    loss_info, reward_info, (total_time) / 60.)
        return info