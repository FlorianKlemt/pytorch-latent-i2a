import os
import torch
from torch.autograd import Variable
import numpy as np
import time
import cv2
import sys
from random import randint
from Environment_Model.load_imagination_core import load_imagination_core
from Environment_Model.load_preprocessed_atari_environment import load_atari_environment
from Environment_Model.environment_model import EMModel_LSTM_One_Reward
from Environment_Model.environment_model import EMModel_used_for_Pong_I2A
from Environment_Model.environment_model import EMModel_Same_Kernel_Size
from Environment_Model.environment_model import PongEM_Big_Model


class RenderImaginationCore():
    def __init__(self):
        render_window_sizes = (400, 400)
        cv2.namedWindow('imagination_core', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('imagination_core', render_window_sizes)
        cv2.namedWindow('start_state', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('start_state', render_window_sizes)

    def render_observation(self, window_name, observation):
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
        time.sleep(0.8)


def play_pong_with_imagination_core(imagination_core, atari_env, use_cuda):
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    render = True
    renderer = RenderImaginationCore()

    for i_episode in range(20):
        observation = atari_env.reset()

        # do 20 random actions to get different start observations
        for i in range(randint(20, 50)):
            atari_env.render()
            observation, reward, done, _ = atari_env.step(atari_env.action_space.sample())

        observation = torch.from_numpy(observation).type(FloatTensor)
        observation = Variable(observation.unsqueeze(0), requires_grad=False)

        # render start state
        if render:
            renderer.render_observation('start_state', observation)

        for t in range(10):

            if render:
                renderer.render_observation('imagination_core', observation)

            observation, reward = imagination_core(observation)
            reward = reward.data.cpu().numpy()
            print(t, "reward", np.max(reward[0], 0))

            # action = np.argmax(action.data, 1)


use_cuda = True
root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))



# big model from felix
# EMModel = PongEM_Big_Model


# ToDo ArgsParser
atari_env = load_atari_environment("PongDeterministic-v4")

# small model which only predicts one reward
EMModel = EMModel_LSTM_One_Reward
imagination_core = load_imagination_core(action_space=atari_env.action_space.n,
                                         policy_model="PongDeterministic-v4_21",
                                         load_policy_model_dir="trained_models/",
                                         EMModel=EMModel,
                                         environment_model_name="pong_em_one_reward",
                                         load_environment_model_dir="trained_models/environment_models/",
                                         root_path=root_dir,
                                         use_cuda=use_cuda)


'''
# model version used for training of pong with i2a on the server
EMModel = EMModel_used_for_Pong_I2A
imagination_core = load_imagination_core(action_space=atari_env.action_space.n,
                                         policy_model="PongDeterministic-v4_21",
                                         load_policy_model_dir="trained_models/IC_Policy/",
                                         EMModel=EMModel,
                                         environment_model_name="small_pong_em_lstm",
                                         load_environment_model_dir="trained_models/environment_models/",
                                         root_path=root_dir,
                                         use_cuda=use_cuda)
'''

play_pong_with_imagination_core(imagination_core, atari_env=atari_env, use_cuda=use_cuda)
