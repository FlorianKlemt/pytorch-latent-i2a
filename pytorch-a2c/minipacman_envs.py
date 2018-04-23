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

#class WrapPyTorchMiniPacman(gym.ObservationWrapper):
#    def __init__(self, env=None, image_size = 19):
#        super(WrapPyTorchMiniPacman, self).__init__(env)
#        self.observation_space = Box(0.0, 1.0, shape=[4, 19, 19], dtype=np.float32)
#
#    def observation(self, observation):
#        return observation

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

class MiniFrameStack(gym.Wrapper):
    def __init__(self, env, num_frames, low = 0., high = 1.):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.num_frames = num_frames
        self.frames = collections.deque(maxlen=self.num_frames)
        shp = env.observation_space.shape
        #assert shp[2] == 1  # can only stack 1-channel frames
        self.observation_space = spaces.Box(low=low, high=high, shape=(num_frames, *shp), dtype=np.float32)

    def reset(self):
        self.frames.clear()
        obs = self.env.reset()
        for i in range(self.num_frames):
            self.frames.append(obs)
        return self.observation()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self.observation(), reward, done, info

    def observation(self):
        return np.stack(self.frames)



class WarpMiniPacmanFrame(gym.ObservationWrapper):
    def __init__(self, env, image_resize_size = 19):
        gym.ObservationWrapper.__init__(self, env)
        self.res = image_resize_size
        self.observation_space = spaces.Box(low=0., high=1., shape=(self.res, self.res), dtype=np.float)
        #self.states_deque = collections.deque(maxlen=self.num_frames)

    def observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        #frame = np.array(Image.fromarray(frame*255), dtype=np.uint8)
        frame = np.array(Image.fromarray(frame), dtype=np.float)
        zeros = np.zeros((19,4))
        frame = np.append(frame, zeros)
        #frame = frame.reshape((19, 19, 1)) #???
        frame = frame.reshape((19, 19))
        return frame

    #def step(self, action):
    #    obs, reward, done, info = self.env.step(action)
    #    obs = self.frame_from_observation(obs)
    #    self.states_deque.append(obs)
    #    return np.stack(self.states_deque), reward, done, info

    #def reset(self):
    #    self.states_deque.clear()
    #    obs = self.env.reset()
    #    obs = self.frame_from_observation(obs)
    #    for i in range(self.num_frames):
    #        self.states_deque.append(obs)
    #    return np.stack(self.states_deque)

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

        env = MiniFrameStack(env, 4)
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

    env = MiniFrameStack(env, 4)
    return env