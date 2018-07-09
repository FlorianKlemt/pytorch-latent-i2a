import gym
import numpy as np
import collections
from PIL import Image
from gym import spaces
from baselines import bench
from baselines.common.atari_wrappers import EpisodicLifeEnv,NoopResetEnv,MaxAndSkipEnv,FireResetEnv,ClipRewardEnv
import os

class MiniFrameStack(gym.Wrapper):
    def __init__(self, env, num_frames, low = 0., high = 1.):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.num_frames = num_frames
        self.frames = collections.deque(maxlen=self.num_frames)
        shp = env.observation_space.shape
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



class WarpMiniPacmanFrameGreyScale(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(low=0., high=1., shape=(1, obs_shape[0], obs_shape[1]), dtype=np.float)
        #self.states_deque = collections.deque(maxlen=self.num_frames)

    def observation(self, obs):
        # convert to grey scale
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        frame = np.array(Image.fromarray(frame), dtype=np.float)
        frame = frame.reshape(self.observation_space.shape)
        return frame

class WarpMiniPacmanFrameRGB(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(low=0., high=1., shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.float)

    def observation(self, obs):
        return obs.reshape(self.observation_space.shape)#obs.transpose(2,0,1)

class WarpMiniPacmanFrameSquared(gym.ObservationWrapper):
    def __init__(self, env, image_resize_size = 19):
        gym.ObservationWrapper.__init__(self, env)
        self.res = image_resize_size
        self.observation_space = spaces.Box(low=0., high=1., shape=(1, self.res, self.res), dtype=np.float)
        #self.states_deque = collections.deque(maxlen=self.num_frames)

    def observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        #frame = np.array(Image.fromarray(frame*255), dtype=np.uint8)
        frame = np.array(Image.fromarray(frame), dtype=np.float)
        zeros = np.zeros((19,4))
        frame = np.append(frame, zeros)
        #frame = frame.reshape((19, 19, 1)) #???
        frame = frame.reshape((1, 19, 19))
        return frame



def make_custom_env(env_id, seed, rank, log_dir, grey_scale):
    def _thunk():
        episodic_life = True
        env = gym.make(env_id)

        env.seed(seed + rank)
        env.frameskip = 1

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        if episodic_life:
            env = EpisodicLifeEnv(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=1)

        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if grey_scale:
            env = WarpMiniPacmanFrameGreyScale(env)
        else:
            env = WarpMiniPacmanFrameRGB(env)
        #env = MiniFrameStack(env, 4)
        return env

    return _thunk


def make_custom_env_squared(env_id, seed, rank, log_dir):
    def _thunk():
        episodic_life = True
        clip_rewards = False     #this should always be false
        image_resize_size = 19

        env = gym.make(env_id)

        env.seed(seed + rank)
        env.frameskip = 1

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        if episodic_life:
            env = EpisodicLifeEnv(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=1)

        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        env = WarpMiniPacmanFrameSquared(env, image_resize_size)
        if clip_rewards:
            env = ClipRewardEnv(env)

        #env = MiniFrameStack(env, 4)
        return env

    return _thunk


#for State Space Encoder
class ClipAtariFrameSizeTo200x160(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.height = 200
        self.width = 160
        self.channels = self.observation_space.shape[0]
        self.observation_space = spaces.Box(low=0., high=1.,
            shape=(self.channels, self.height, self.width), dtype=np.float)

    def observation(self, frame):
        return frame[:, :self.height, :self.width]

class RewardScaling(gym.Wrapper):
    def __init__(self, env, scaling_factor=0.1):    #0.1 is good for MsPacman
        gym.Wrapper.__init__(self, env)
        self.scaling_factor = scaling_factor

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        reward *= self.scaling_factor
        return obs, reward, done, info

class NegativeRewardForDying(gym.Wrapper):
    def __init__(self, env, reward_for_dying = -100):
        gym.Wrapper.__init__(self, env)
        self.lives = self.env.unwrapped.ale.lives()
        self.reward_for_dying = reward_for_dying

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives:
            reward += self.reward_for_dying

        self.lives = self.unwrapped.ale.lives()
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs