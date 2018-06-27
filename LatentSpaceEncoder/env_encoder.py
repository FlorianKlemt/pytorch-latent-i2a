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
    args_parser.add_argument('--grey-scale', action='store_true', default=False,
                             help='True to convert to grey_scale images')
    args_parser.add_argument('--save-interval', type=int, default=10,
                             help='Save interval for auto_encoder and env_encoder (default: 10)')
    args_parser.add_argument('--save-model-path', default='/home/flo/Dokumente/I2A_GuidedResearch/pytorch-a2c-ppo-acktr/LatentSpaceEncoder/trained_autoencoder_models/', help='path to save and load models')
    args_parser.add_argument('--test', action='store_true', default=False, help='true to run test_encoder parallel')
    args_parser.add_argument('--no-cuda', action='store_true', default=False, help='true to compute on cpu')
    args_parser.add_argument('--cnn-model', action='store_true', default=False,
                             help='True to use large cnn model, False to use small linear model (currently: False for MiniPacman'
                                  'True for MsPacman)')
    args_parser.add_argument('--frame-loss-weight', type=float, default=1, help='factor to multiply frame-loss with')
    args_parser.add_argument('--reward-loss-weight', type=float, default=1, help='factor to multiply reward-loss with')
    args_parser.add_argument('--latent-space', type=int, default=128,
                             help='size of the latent space (default: 128)  (NOTE: when using the CNN model the latent-space dim is'
                                  'calculated automatically overwriting this value)')
    args_parser.add_argument('--encoder-space', type=int, default=256,
                             help='size of the encoder space (default: 256)')
    args_parser.add_argument('--hidden-space', type=int, default=512,
                             help='size of the hidden space (default: 512)  (NOTE: this value only has meaning when using the LinearModel)')
    args_parser.add_argument('--auto-encoder-lr', type=float, default=1e-4, help='autoencoder learning rate (default: 1e-4)')
    args_parser.add_argument('--auto-encoder-weight-decay', type=float, default=1e-2, help='autoencoder weight decay (default: 1e-2)')
    args_parser.add_argument('--env-encoder-lr', type=float, default=1e-4, help='env encoder learning rate (default: 1e-4)')
    args_parser.add_argument('--env-encoder-weight-decay', type=float, default=1e-2, help='env encoder weight decay (default: 1e-2)')
    args = args_parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.test:
        import multiprocessing as mp
        mp.set_start_method('spawn')    #get the exorcist, this method is evil

    #magic do not change
    plt.switch_backend('TKAgg')
    plt.ion()

    use_cuda = True
    from a2c_models.i2a_mini_model import I2A_MiniModel
    from a2c_models.a2c_policy_wrapper import A2C_PolicyWrapper

    #create environment to train on
    if "MiniPacman" in args.env:
        env = make_custom_env(args.env, seed=1, rank=0, log_dir=None, grey_scale=args.grey_scale)()
        if not args.use_only_last_frame_of_state:
            env = EncoderFrameStack(env, 4)
    else:
        env = make_env(args.env, seed=1, rank=0, log_dir=None, grey_scale=args.grey_scale, stack_frames=not args.use_only_last_frame_of_state)()

    obs_shape = env.observation_space.shape

    #num_autoencoder_inputs = (1 if args.use_only_last_frame_of_state else obs_shape[0])   #this is the batch dimension (the frame stack)
    num_autoencoder_inputs = obs_shape[0]

    if args.cnn_model:
        auto_encoder_model = CNNAutoEncoderModel(num_inputs=num_autoencoder_inputs, latent_space=args.latent_space,
                                                 input_size=obs_shape[1:])
    else:
        auto_encoder_model = LinearAutoEncoderModel(num_inputs = num_autoencoder_inputs, input_size=obs_shape[1:],
                                                 latent_space=args.latent_space, hidden_space=args.hidden_space)
    if use_cuda:
        auto_encoder_model.cuda()

    action_space = env.action_space.n
    #in the i2a_minimodel framstack and rgb channels are thrown together as channels
    policy = A2C_PolicyWrapper(I2A_MiniModel(obs_shape=obs_shape, action_space=action_space, use_cuda=use_cuda))
    if use_cuda:
        policy.cuda()

    latent_space = auto_encoder_model.latent_space_dim
    env_encoder_model = EnvEncoderModel(num_inputs=1, latent_space=latent_space, encoder_space=args.encoder_space,
                                        action_broadcast_size=10, use_cuda=use_cuda)
    if use_cuda:
        env_encoder_model.cuda()

    loss_criterion = torch.nn.MSELoss()
    #auto_optimizer = torch.optim.RMSprop(auto_encoder_model.parameters(), lr=0.00005, weight_decay=1e-5)             #0.0001!!!
    #env_encoder_optimizer = torch.optim.RMSprop(env_encoder_model.parameters(), lr=0.00005, weight_decay=1e-5)       #0.0001!!!
    auto_optimizer = torch.optim.RMSprop(auto_encoder_model.parameters(), lr=args.auto_encoder_lr, weight_decay=args.auto_encoder_weight_decay)
    env_encoder_optimizer = torch.optim.RMSprop(env_encoder_model.parameters(), lr=args.env_encoder_lr, weight_decay=args.env_encoder_weight_decay)

    latent_space_trainer = LatentSpaceEnvModelTrainer(auto_encoder_model=auto_encoder_model, env_encoder_model=env_encoder_model,
                                                      loss_criterion=loss_criterion, auto_optimizer=auto_optimizer,
                                                      next_pred_optimizer=env_encoder_optimizer, use_cuda=use_cuda,
                                                      visualize=True,
                                                      args=args)

    if args.test:
        import copy
        from LatentSpaceEncoder.test_encoder import TestLatentSpaceModel
        test_process = TestLatentSpaceModel(env=env,
                                            auto_encoder=copy.deepcopy(auto_encoder_model),
                                            env_encoder=copy.deepcopy(env_encoder_model),
                                            auto_encoder_load_path=args.save_model_path + "autoencoder.pt",
                                            env_encoder_load_path=args.save_model_path + "envencoder.pt",
                                            rollout_policy=policy,
                                            args=args)

    if args.train_batchwise:
        latent_space_trainer.train_env_encoder_batchwise(env, policy, use_cuda)
    else:
        latent_space_trainer.train_env_encoder(env, policy, use_cuda)

    if args.test:
        test_process.stop()




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




def make_env(env_id, seed, rank, log_dir, grey_scale, stack_frames):
    from baselines import bench
    from baselines.common.atari_wrappers import make_atari, wrap_deepmind, EpisodicLifeEnv, ClipRewardEnv
    import os
    from envs import WrapPyTorch
    def _thunk():
        env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = custom_make_atari(env_id)
        env.seed(seed + rank)
        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        if is_atari:
            env = EpisodicLifeEnv(env)
            env = ClipRewardEnv(env)
            #env = wrap_deepmind(env)
        env = FrameUIntToFloat(env)
        if grey_scale:
            env = WarpFrameGrayScale(env)
        else:
            env = ReshapeRGBChannels(env)
        if stack_frames:
            env = EncoderFrameStack(env, 4)
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        #obs_shape = env.observation_space.shape
        #if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        #    env = WrapPyTorch(env)

        return env

    return _thunk


def custom_make_atari(env_id):
    from baselines.common.atari_wrappers import NoopResetEnv
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    #env = MaxAndSkipEnv(env, skip=4)   #TODO: maybe 2 skip
    return env


def create_atari_environment(env_id, seed, rank, log_dir, grey_scale, stack_frames):
    from baselines import bench
    from baselines.common.atari_wrappers import EpisodicLifeEnv, ClipRewardEnv
    import os
    env = gym.make(env_id)
    is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
    if is_atari:
        env = custom_make_atari(env_id)
    env.seed(seed + rank)
    if log_dir is not None:
        env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
    if is_atari:
        env = EpisodicLifeEnv(env)
        env = ClipRewardEnv(env)
    env = FrameUIntToFloat(env)
    if grey_scale:
        env = WarpFrameGrayScale(env)
    else:
        env = ReshapeRGBChannels(env)
    if stack_frames > 1:
        env = EncoderFrameStack(env, stack_frames)
    return env

def make_env_ms_pacman(env_id, seed, rank, log_dir, grey_scale, stack_frames):
    from custom_envs import ClipAtariFrameSizeTo200x160
    def _thunk():
        env = create_atari_environment(env_id=env_id, seed= seed, rank=rank, log_dir=log_dir, grey_scale=grey_scale, stack_frames=stack_frames)
        env = ClipAtariFrameSizeTo200x160(env=env)
        return env

    return _thunk


if __name__ == '__main__':
    main()