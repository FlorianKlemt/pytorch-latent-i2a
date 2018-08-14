import gym

import collections
from gym import spaces
class EncoderFrameStack(gym.Wrapper):
    def __init__(self, env, num_frames, low = 0., high = 1.):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.num_frames = num_frames
        self.frames = collections.deque(maxlen=self.num_frames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=low, high=high, shape=(num_frames*shp[0], *shp[1:]), dtype=np.float32)
        #self.observation_space = spaces.Box(low=low, high=high, shape=(num_frames, *shp), dtype=np.float32)

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
        return np.concatenate(self.frames)  #here we use concatenate instead of stack

import numpy as np
class FrameUIntToFloat(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        box = self.observation_space
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(box.shape))

    def _observation(self, obs):
        frame = obs / 255.
        frame[frame < 0] = 0.
        frame[frame > 1.] = 1.
        return frame


class WarpFrameGrayScale(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        box = self.observation_space
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(box.shape[0], box.shape[1]))

    def _observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        return frame.reshape((frame.shape[0], frame.shape[1]))


class ReshapeRGBChannels(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        box = self.observation_space
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(box.shape[2], box.shape[0], box.shape[1]))

    def _observation(self, obs):
        return obs.transpose(2,0,1)


class SkipFramesEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip       = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

