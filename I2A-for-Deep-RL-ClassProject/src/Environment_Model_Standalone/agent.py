import os
import numpy as np
import cv2
import utils
import logging
import pickle


class Agent(object):
    def __init__(self):
        self.logger = logging.getLogger('main')

    def play_step(self):
        """
        Perform one step in the game (select action, execute action) and report back

        :return:
        - current game state (frame) as numpy array (for example)
        - action that was performed on game state
        - next game state (frame) that was the result of the action performed
        - reward the agent got as a result of the action performed as a scalar (float)
        """
        raise NotImplementedError("not implemented in base class")


class OverfitOnRandomFramesAgent(Agent):

    def __init__(self, num_frames=1, game_state_frame_dim=(1, 80, 80), game_actions=6):
        super(OverfitOnRandomFramesAgent, self).__init__()
        self.num_frames = num_frames
        self.num_game_actions = game_actions
        self.frames = np.random.rand(num_frames, *game_state_frame_dim)
        self.rewards = np.random.rand(num_frames)

        # the actions are created as a one hot vector one for each frame
        self.actions = np.eye(game_actions)[np.random.choice(game_actions, num_frames)]

        self.internal_game_state_index = 0

    def play_step(self):
        # get current frame from (1, 80, 80) to (1, 1, 80, 80)
        current_frame = np.expand_dims(self.frames[self.internal_game_state_index], axis=0)
        action = self.actions[self.internal_game_state_index]

        # start from the beginning again when we reach the end
        self.internal_game_state_index = (self.internal_game_state_index + 1) % self.actions.shape[0]

        next_reward = self.rewards[self.internal_game_state_index]
        next_frame = np.expand_dims(self.frames[self.internal_game_state_index], axis=0)

        # return as numpy array (even scalars)
        return current_frame, action, np.array([next_reward]), next_frame


class OverfitOnSinglePongFrameAgent(OverfitOnRandomFramesAgent):

    def __init__(self):
        super(OverfitOnSinglePongFrameAgent, self).__init__(1, (1, 80, 80), 6)
        dataset_path = os.path.join(os.getcwd(), 'datasets')

        pong_test_image_path = os.path.join(dataset_path, 'pong_test_image.png')

        # load image and make sure it's scaled to [0, 1]
        frame = cv2.imread(pong_test_image_path, cv2.IMREAD_GRAYSCALE)
        frame_min = frame.min()
        frame_max = frame.max()
        frame = (frame - frame_min) / (frame_max - frame_min + 0.0000000001)
        self.frames = np.array([[frame]])


class OverfitOnNFramesAgent(OverfitOnRandomFramesAgent):

    def __init__(self, dataset_name, num_actions, n_frames=400):
        super(OverfitOnNFramesAgent, self).__init__(n_frames, (1, 80, 80), num_actions)

        # taken from an A3C agent playing pong
        frames = []

        # validate dataset
        dataset_path = os.path.join(os.getcwd(), 'datasets', dataset_name)
        if utils.check_if_file_exists(os.path.join(dataset_path, 'actions_and_rewards.pkl')) and utils.check_if_file_exists(os.path.join(dataset_path, 'frame_0.png')):
            # dataset ok
            self.logger.debug('Dataset ok.')

            # Load actions and rewards
            actions = []
            rewards = []
            with open(os.path.join(dataset_path, 'actions_and_rewards.pkl'), 'rb') as f:
                actions, rewards = pickle.load(f)

            # validate rewards. We need as many rewards as actions
            if len(actions) == len(rewards):

                if len(actions) >= self.num_frames:

                    self.logger.debug('Loading {0} frames.'.format(self.num_frames))
                    for i in range(self.num_frames):
                        test_image_path = os.path.join(dataset_path, 'frame_{0}.png'.format(i))
                        frame = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

                        frame_min = frame.min()
                        frame_max = frame.max()
                        frame = (frame - frame_min) / (frame_max - frame_min + 0.0000000001)
                        frame *= 255
                        frame = frame.astype(np.uint8)
                        frames.append([frame])
                        self.rewards[i] = rewards[i]

                        # convert action to one-hot
                        action = np.zeros(self.num_game_actions)
                        action[actions[i]] = 1
                        self.actions[i] = action
                    self.frames = np.array(frames)
                    self.logger.info('Dataset is loaded. Frames shape: ' + str(self.frames.shape))
                else:
                    self.logger.warning('More frames than actions!')
            else:
                self.logger.warning('Number of actions and rewards is not the same')
        else:
            self.logger.error('Dataset is not valid')
