import torch
from LatentSpaceEncoder.EnvEncoderModel import EnvEncoderModel
from LatentSpaceEncoder.AutoEncoderModel import AutoEncoderModel, LinearAutoEncoderModel, CNNAutoEncoderModel
from LatentSpaceEncoder.LatentSpaceEnvModelTrainer import LatentSpaceEnvModelTrainer
from custom_envs import make_custom_env
import matplotlib.pyplot as plt
import cv2
import gym
import gym_minipacman
import argparse


def main():
    args_parser = argparse.ArgumentParser(description='LatentSpaceEncoder')
    args_parser.add_argument('--env', default='RegularMiniPacmanNoFrameskip-v0', help='environment to use')
    args_parser.add_argument('--use_only_last_frame_of_state', action='store_true', default=False,
                             help='use only last frame of state as input to the autoencoder')
    args_parser.add_argument('--train-batchwise', action='store_true', default=False,
                             help='train autoencoder and envencoder on frames sampled from a deque (conceptwise like a replay memory),'
                                  'helps to reduce the bias on specific image orders, as they occur in a live game')
    args = args_parser.parse_args()

    #magic do not change
    plt.switch_backend('TKAgg')
    plt.ion()

    use_cuda = True
    from A2C_Models.I2A_MiniModel import I2A_MiniModel
    from A2C_Models.A2C_PolicyWrapper import A2C_PolicyWrapper

    latent_space = 32#64
    encoder_space = 128#128
    hidden_space = 128#128

    #create environment to train on
    if "MiniPacman" in args.env:
        #TODO: currently this does not have a framestack -> state is only one frame
        env = make_custom_env(args.env, seed=1, rank=0, log_dir=None)()
    else:
        env = make_env(args.env, seed=1, rank=0, log_dir=None)()

    obs_shape = env.observation_space.shape

    #num_autoencoder_inputs = (1 if args.use_only_last_frame_of_state else 4) * 3    #TODO: this assumes rgb image with 3 channels
    num_autoencoder_inputs = obs_shape[0]   #this is the batch dimension (the frame stack)
    auto_encoder_model = LinearAutoEncoderModel(num_inputs = num_autoencoder_inputs, input_size=obs_shape[1:],
                                                latent_space=latent_space, hidden_space=hidden_space)
    if use_cuda:
        auto_encoder_model.cuda()

    action_space = env.action_space.n
    policy = A2C_PolicyWrapper(I2A_MiniModel(obs_shape=obs_shape, action_space=action_space, use_cuda=use_cuda))
    if use_cuda:
        policy.cuda()

    env_encoder_model = EnvEncoderModel(num_inputs=1, latent_space=latent_space, encoder_space=encoder_space,
                                        action_broadcast_size=10, use_cuda=use_cuda)
    if use_cuda:
        env_encoder_model.cuda()
    loss_criterion = torch.nn.MSELoss()
    auto_optimizer = torch.optim.RMSprop(auto_encoder_model.parameters(), lr=0.00005, weight_decay=1e-5)             #0.0001!!!
    env_encoder_optimizer = torch.optim.RMSprop(env_encoder_model.parameters(), lr=0.00005, weight_decay=1e-5)       #0.0001!!!

    latent_space_trainer = LatentSpaceEnvModelTrainer(auto_encoder_model=auto_encoder_model, env_encoder_model=env_encoder_model,
                                                      loss_criterion=loss_criterion, auto_optimizer=auto_optimizer,
                                                      next_pred_optimizer=env_encoder_optimizer, use_cuda=use_cuda,
                                                      visualize=True,
                                                      use_only_last_frame_of_state=args.use_only_last_frame_of_state)
    if args.train_batchwise:
        latent_space_trainer.train_env_encoder_batchwise(env, policy, use_cuda)
    else:
        latent_space_trainer.train_env_encoder(env, policy, use_cuda)





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

import collections
from gym import spaces
class EncoderFrameStack(gym.Wrapper):
    def __init__(self, env, num_frames, low = 0., high = 1.):
        """Buffer observations and stack across channels (last axis)."""
        gym.Wrapper.__init__(self, env)
        self.num_frames = num_frames
        self.frames = collections.deque(maxlen=self.num_frames)
        shp = env.observation_space.shape
        #self.observation_space = spaces.Box(low=low, high=high, shape=(num_frames*shp[0], *shp[1:]), dtype=np.float32)
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
        #return np.concatenate(self.frames)  #here we use concatenate instead of stack


def make_env(env_id, seed, rank, log_dir):
    from baselines import bench
    from baselines.common.atari_wrappers import make_atari, wrap_deepmind, EpisodicLifeEnv, ClipRewardEnv
    import os
    from envs import WrapPyTorch
    def _thunk():
        env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        if is_atari:
            env = EpisodicLifeEnv(env)
            env = ClipRewardEnv(env)
            #env = wrap_deepmind(env)
        env = FrameUIntToFloat(env)
        #env = WarpFrameGrayScale(env)
        env = ReshapeRGBChannels(env)
        #env = EncoderFrameStack(env, 4)
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        #obs_shape = env.observation_space.shape
        #if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        #    env = WrapPyTorch(env)

        return env

    return _thunk

if __name__ == '__main__':
    main()