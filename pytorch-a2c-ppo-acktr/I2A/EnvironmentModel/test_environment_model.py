import os
import torch
from torch.autograd import Variable
import numpy as np
import time
import cv2
import sys
from random import randint
import collections
from collections import deque
from custom_envs import make_custom_env
from I2A.EnvironmentModel.MiniPacmanEnvModel import MiniPacmanEnvModel
from I2A.ImaginationCore import ImaginationCore
from A2C_Models.I2A_MiniModel import I2A_MiniModel
from A2C_Models.A2C_PolicyWrapper import A2C_PolicyWrapper
import gym_minipacman

class RenderImaginationCore():
    def __init__(self):
        render_window_sizes = (400, 400)
        cv2.namedWindow('imagination_core', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('imagination_core', render_window_sizes)
        cv2.namedWindow('start_state', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('start_state', render_window_sizes)

    def render_observation(self, window_name, observation):
        drawable_state = (observation.data[0]).unsqueeze(0)
        #drawable_state = (observation.data[0][-1]).unsqueeze(0)
        drawable_state = np.swapaxes(drawable_state, 0, 1)
        drawable_state = np.swapaxes(drawable_state, 1, 2)
        drawable_state = drawable_state.numpy()

        frame_data = (drawable_state * 255.0)
        #frame_data = drawable_state

        frame_data[frame_data < 0] = 0
        frame_data[frame_data > 255] = 255
        frame_data = frame_data.astype(np.uint8)

        cv2.imshow(window_name, frame_data)
        cv2.waitKey(1)
        time.sleep(1.)


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
            observation = Variable(observation, requires_grad=False)
            states.append(observation)

        # render start state
        if render:
            renderer.render_observation('start_state', observation)

        for t in range(5):
            if render:
                render_obs = observation if observation.data.shape[1]==19 or observation.data.shape[1]==15 else observation[-1][-1].unsqueeze(0) #kill me now
                renderer.render_observation('imagination_core', render_obs)
            stacked_frames = torch.stack(states, dim=1)
            action = imagination_core.sample(stacked_frames[:,-1:])
            observation, reward = imagination_core(stacked_frames, action)
            states.append(observation[0,-1:])
            reward = reward.data.cpu().numpy()
            print(t, "reward", np.max(reward[0], 0))

            # action = np.argmax(action.data, 1)

### Begin Main ###
use_cuda = True
#root_dir = '/home/flo/Dokumente/I2A_GuidedResearch/pytorch-a2c-ppo-acktr/trained_models/environment_models/'    #os.path.dirname(os.path.realpath(sys.argv[0]))


env_name = "RegularMiniPacmanNoFrameskip-v0"
env = make_custom_env(env_name, seed=1, rank=1, log_dir=None, grey_scale=True)()

# small model which only predicts one reward
EMModel = MiniPacmanEnvModel

environment_model = EMModel(obs_shape=env.observation_space.shape,
                                num_actions=env.action_space.n,
                                reward_bins=env.unwrapped.reward_bins,
                                use_cuda=use_cuda)
saved_state = torch.load("../../trained_models/environment_models/RegularMiniPacman_EnvModel_0.dat", map_location=lambda storage, loc: storage)
environment_model.load_state_dict(saved_state)

rollout_policy = A2C_PolicyWrapper(I2A_MiniModel(obs_shape=env.observation_space.shape, action_space=env.action_space.n, use_cuda=use_cuda))
if use_cuda:
    environment_model.cuda()
    rollout_policy.cuda()

imagination_core = ImaginationCore(env_model=environment_model, rollout_policy=rollout_policy)
#imagination_core = ImaginationCore(num_inputs=4, action_space=env.action_space.n,
#                                             em_model_reward_bins=em_model_reward_bins, use_cuda=use_cuda, require_grad=False)    #no policy grads required for this

play_with_imagination_core(imagination_core, env=env, use_cuda=use_cuda)
