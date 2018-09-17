import gym
from gym import spaces
import numpy as np

from gym_envs.envs_wrapper import SkipFramesEnv, \
    FrameUIntToFloat, WarpFrameGrayScale,\
    ReshapeRGBChannels, EncoderFrameStack


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

def custom_make_atari(env_id, skip_frames = 1):
    from baselines.common.atari_wrappers import NoopResetEnv
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = SkipFramesEnv(env, skip=skip_frames)   #TODO: maybe 2 skip
    return env


def create_atari_environment(env_id, seed, rank, log_dir, grey_scale, stack_frames, skip_frames):
    from baselines import bench
    import os
    env = gym.make(env_id)
    is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
    if is_atari:
        env = custom_make_atari(env_id, skip_frames=skip_frames)
    env.seed(seed + rank)
    if log_dir is not None:
        env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
    env = FrameUIntToFloat(env)
    if grey_scale:
        env = WarpFrameGrayScale(env)
    else:
        env = ReshapeRGBChannels(env)
    if stack_frames > 1:
        env = EncoderFrameStack(env, stack_frames)
    return env

def make_env_ms_pacman(env_id, seed, rank, log_dir, grey_scale, stack_frames, skip_frames):
    def _thunk():
        env = create_atari_environment(env_id=env_id, seed= seed, rank=rank,
                                       log_dir=log_dir, grey_scale=grey_scale,
                                       stack_frames=stack_frames, skip_frames=skip_frames)
        env = ClipAtariFrameSizeTo200x160(env=env)
        env = NegativeRewardForDying(env, -100)
        env = RewardScaling(env, 0.1)
        return env

    return _thunk
