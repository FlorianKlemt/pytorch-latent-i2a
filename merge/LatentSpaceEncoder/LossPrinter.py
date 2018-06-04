import numpy as np
import matplotlib.pyplot as plt

class LossPrinter():
    def __init__(self):
        self.auto_loss_list = []
        self.latent_pred_loss_list = []
        self.reset()
        self.auto_loss_plot, self.latent_pred_loss_plot = self.init_loss_plot()

    def reset(self):
        self.game_step_counter = 0
        self.total_game_auto_loss = 0
        self.total_game_pred_loss = 0

    def add_loss(self, first_state_loss, latent_loss):
        self.auto_loss_list.append(first_state_loss)
        self.latent_pred_loss_list.append(latent_loss)
        self.total_game_auto_loss += first_state_loss
        self.total_game_pred_loss += latent_loss
        self.game_step_counter += 1

    def print_episode(self, i_episode):
        print("Episode ", i_episode, " auto_loss: ", self.total_game_auto_loss / self.game_step_counter, " pred_loss: ",
              self.total_game_pred_loss / self.game_step_counter)

    def plot_loss(self, i_episode):
        self.plot_smooth_loss(i_episode, self.auto_loss_list, self.auto_loss_plot)
        self.plot_smooth_loss(i_episode, self.latent_pred_loss_list, self.latent_pred_loss_plot)

    def plot_smooth_loss(self, episode, loss_list, subplot):
        subplot.cla()
        k = list(range(0, episode + 1))
        plot_list = [0] * len(loss_list)
        for i in range(len(loss_list)):
            plot_list[i] = np.mean(loss_list[max(i - 5, 0):min(i + 5, len(loss_list))])
        subplot.plot(k, plot_list, 'r')
        plt.show()
        plt.pause(0.001)

    def plot_non_smooth_loss(self, episode, loss_list, subplot):
        k = list(range(0, episode + 1))
        subplot.plot(k, loss_list, 'r')
        plt.show()
        plt.pause(0.001)

    def init_loss_plot(self):
        fig = plt.figure()

        auto_loss_plot = plt.subplot(121)
        auto_loss_plot.set_yscale('log')  # gca for get_current_axis
        plt.xlabel('Number of Episodes')
        plt.ylabel('Smoothed Loss')
        plt.autoscale(enable=True, axis='x', tight=None)
        plt.title("Autoencoder Loss")
        plt.legend(loc=4)

        latent_pred_loss_plot = plt.subplot(122)
        latent_pred_loss_plot.set_yscale('log')  # gca for get_current_axis
        plt.xlabel('Number of Episodes')
        plt.ylabel('Latent Space Prediction Loss')
        plt.autoscale(enable=True, axis='x', tight=None)
        plt.title("MiniPacman")
        plt.legend(loc=4)
        return auto_loss_plot, latent_pred_loss_plot