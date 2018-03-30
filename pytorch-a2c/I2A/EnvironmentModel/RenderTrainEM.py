import os
import time
import cv2
import logging
import numpy as np

class RenderTrainEM():
    def __init__(self, log_name, delete_log_file = True):
        render_window_sizes = (400, 400)
        cv2.namedWindow('target', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('target', render_window_sizes)
        cv2.namedWindow('prediction', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('prediction', render_window_sizes)
        #cv2.namedWindow('differnece', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('differnece', render_window_sizes)

        # this logger will both print to the console as well as the file

        log_file = os.path.join('logs', 'em_trainer_' + log_name +'.log')
        #log_file = 'logs/em_trainer_log.log'
        if delete_log_file == True and os.path.exists(log_file):
            os.remove(log_file)
        self.logger_prediction = logging.getLogger('em_trainer_log')
        self.logger_prediction.addHandler(logging.FileHandler(log_file))
        self.logger_prediction.setLevel(logging.INFO)
        self.iteration = 0
        self.start_time = time.time()
        self.train_total_time = 0

    def render_observation_in_window(self, window_name, observation):
        drawable_state = observation.data[0]
        drawable_state = np.swapaxes(drawable_state, 0, 1)
        drawable_state = np.swapaxes(drawable_state, 1, 2)

        drawable_state = drawable_state.numpy()

        frame_data = (drawable_state * 255.0)

        frame_data[frame_data < 0] = 0
        frame_data[frame_data > 255] = 255
        frame_data = frame_data.astype(np.uint8)

        cv2.imshow(window_name, frame_data)
        cv2.waitKey(1)


    def render_observation(self, target, prediction):
        self.render_observation_in_window('target', target)
        self.render_observation_in_window('prediction', prediction)
        #loss = (target - prediction) + 0.5
        #self.render_observation_in_window('differnece', loss)

    def log_loss_and_reward(self, episode, state_loss, reward_loss,
                            reward_prediction, reward):
        self.iteration += 1
        duration = time.time() - self.start_time
        self.start_time = time.time()
        self.train_total_time += duration

        state_loss = state_loss.data.cpu().numpy()[0]
        reward_loss = reward_loss.data.cpu().numpy()[0]
        reward_prediction = reward_prediction.data.cpu().numpy()[0][0]
        reward = reward.data.cpu().numpy()[0]

        summary = self.get_step_summary(episode, self.iteration,
                                   state_loss, reward_loss,
                                   reward_prediction, reward,
                                   duration, self.train_total_time)
        self.logger_prediction.info(summary)
        if self.iteration % 10 == 0 or reward < -0.9 or reward > 0.9:
            print(summary)

    def get_step_summary(episode, iteration,
                         frame_loss, reward_loss,
                         pred_reward, reward,
                         duration, total_time):

        iteration_suffix = ''
        if iteration >= 1000:
            iteration /= 1000
            iteration_suffix = 'k'

        str_game = 'Episode: {0:.0f}\t'.format(episode)
        str_iteration = 'it: {0:.1f}{1}\t\t'.format(iteration, iteration_suffix)
        str_loss = 'frame loss: {0:.9f}\t reward loss: {1:.5f}\t\t'.format(frame_loss, reward_loss)
        str_reward = 'pred reward: {0:.3f}\t reward: {1:.3f}\t\t'.format(pred_reward, reward)
        str_time = 'duration: {0:.3f}s\t total duration{1:.3f}min'.format(duration, total_time / 60)
        return str_game + str_iteration + str_loss + str_reward + str_time