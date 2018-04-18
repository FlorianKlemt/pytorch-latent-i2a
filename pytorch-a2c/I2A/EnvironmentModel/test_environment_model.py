import os
import torch
from torch.autograd import Variable
import numpy as np
import time
import cv2
import sys
from random import randint
import collections
from minipacman_envs import make_minipacman_env_no_log
from I2A.EnvironmentModel.MiniPacmanEnvModel import MiniPacmanEnvModel
from I2A.EnvironmentModel.make_environment_model import load_em_model, load_policy
from I2A.ImaginationCore import ImaginationCore, MiniPacmanImaginationCore

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


def play_with_imagination_core(imagination_core, env, use_cuda):
    num_stack = 4
    states = collections.deque(maxlen=num_stack)

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    render = True
    renderer = RenderImaginationCore()

    for i_episode in range(20):
        observation = env.reset()

        # do 20 random actions to get different start observations
        for i in range(randint(20, 50)):
            env.render()
            observation, reward, done, _ = env.step(env.action_space.sample())

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


def load_imagination_core(action_space,
                          policy_model,
                          load_policy_model_dir,
                          EMModel,
                          environment_model_name,
                          load_environment_model_dir,
                          root_path,
                          use_cuda):
    load_policy_model_dir = os.path.join(root_path, load_policy_model_dir)
    load_environment_model_dir = os.path.join(root_path, load_environment_model_dir)

    em_model = load_em_model(EMModel=EMModel,
                             load_environment_model_dir= load_environment_model_dir,
                             environment_model_name= environment_model_name,
                             action_space=action_space,
                             use_cuda=use_cuda)
    policy = load_policy(load_policy_model_dir,
                         policy_model,
                         action_space=action_space,
                         use_cuda=use_cuda)

    if use_cuda:
        em_model.cuda()
    return ImaginationCore(em_model, policy, use_cuda)




### Begin Main ###
use_cuda = True
root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))


env_name = "RegularMiniPacmanNoFrameskip-v0"
env = make_minipacman_env_no_log(env_name)

# small model which only predicts one reward
EMModel = MiniPacmanEnvModel
#imagination_core = load_imagination_core(action_space=env.action_space.n,
#                                         policy_model="PongDeterministic-v4_21",
#                                         load_policy_model_dir="trained_models/",
#                                         EMModel=EMModel,
#                                         environment_model_name="pong_em_one_reward",
#                                         load_environment_model_dir="trained_models/environment_models/",
#                                         root_path=root_dir,
#                                         use_cuda=use_cuda)

imagination_core = MiniPacmanImaginationCore(num_inputs=4, use_cuda=use_cuda)

play_with_imagination_core(imagination_core, env=env, use_cuda=use_cuda)
