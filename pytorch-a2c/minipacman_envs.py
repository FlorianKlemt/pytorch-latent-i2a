import gym
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import *
import numpy as np
from PIL import Image
from gym import spaces
import os
import collections

import torch

class WrapPyTorchMiniPacman(gym.ObservationWrapper):
    def __init__(self, env=None, image_size = 19):
        super(WrapPyTorchMiniPacman, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, shape=[4, 19, 19], dtype=np.float32)

    def observation(self, observation):
        #print(observation.shape)
        return observation
        #return observation.transpose(3, 0, 1, 2)

'''class WarpMiniPacmanFrame(gym.ObservationWrapper):
    def __init__(self, env, image_resize_size = 19):
        gym.ObservationWrapper.__init__(self, env)
        self.res = image_resize_size
        self.observation_space = spaces.Box(low=0, high=255, shape=(19, 19, 1), dtype=np.uint8)

    def observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        #frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
        #    resample=Image.BILINEAR), dtype=np.uint8)
        #frame = np.array(Image.fromarray(frame), dtype=np.float32)
        frame = np.array(Image.fromarray(frame*255), dtype=np.uint8)
        zeros = np.zeros((19,4))
        frame = np.append(frame, zeros)
        frame = frame.reshape((19, 19, 1))
        return frame'''

class WarpMiniPacmanFrame(gym.ObservationWrapper):
    def __init__(self, env, image_resize_size = 19, num_frames=4):
        gym.ObservationWrapper.__init__(self, env)
        self.num_frames = num_frames
        self.res = image_resize_size
        self.observation_space = spaces.Box(low=0., high=1., shape=(19, 19, 1), dtype=np.float)
        self.states_deque = collections.deque(maxlen=self.num_frames)

    def frame_from_observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        #frame = np.array(Image.fromarray(frame*255), dtype=np.uint8)
        frame = np.array(Image.fromarray(frame), dtype=np.float)
        zeros = np.zeros((19,4))
        frame = np.append(frame, zeros)
        #frame = frame.reshape((19, 19, 1)) #???
        frame = frame.reshape((19, 19))
        return frame

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.frame_from_observation(obs)
        self.states_deque.append(obs)
        return np.stack(self.states_deque), reward, done, info

    def reset(self):
        self.states_deque.clear()
        obs = self.env.reset()
        obs = self.frame_from_observation(obs)
        for i in range(self.num_frames):
            self.states_deque.append(obs)
        return np.stack(self.states_deque)

    #def states_to_torch(self, states, use_cuda):
    #    states = np.stack(states)
    #    states = torch.from_numpy(states).float()
    #    states = states.permute(1, 0, 2, 3)
    #    if use_cuda:
    #        states = states.type(torch.cuda.FloatTensor)
    #        states.cuda()
    #    states = Variable(states)  # , requires_grad=False)
    #    return states


def make_minipacman_env(env_id, seed, rank, log_dir):
    def _thunk():
        episode_life = True
        clip_rewards = True
        image_resize_size = 19

        env = gym.make(env_id)

        #magic
        env.frameskip = 1
        #env.unwrapped.ale.setFloat('repeat_action_probability', 0.)

        env.seed(seed + rank)
        env = bench.Monitor(env,
                            os.path.join(log_dir,
                                         "{}.monitor.json".format(rank)))

        if episode_life:
            env = EpisodicLifeEnv(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=1)

        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        env = WarpMiniPacmanFrame(env, image_resize_size)
        if clip_rewards:
            env = ClipRewardEnv(env)

        env = WrapPyTorchMiniPacman(env)
        return env

    return _thunk


def make_minipacman_env_no_log(env_id):
    episode_life = True
    clip_rewards = True
    image_resize_size = 19

    env = gym.make(env_id)

    #magic
    env.frameskip = 1

    if episode_life:
        env = EpisodicLifeEnv(env)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=1)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = WarpMiniPacmanFrame(env, image_resize_size)
    if clip_rewards:
        env = ClipRewardEnv(env)

    env = WrapPyTorchMiniPacman(env)
    return env