import torch
import gym
from environment_model.LatentSpaceEncoder.models_from_paper.state_space_model import SSM

import argparse
from a2c_models.a2c_policy_wrapper import A2C_PolicyWrapper
from a2c_models.atari_model import AtariModel

from gym_envs.env_ms_pacman import make_env_ms_pacman
from rl_visualization.environment_model.test_environment_model import TestEnvironmentModel
import copy
import multiprocessing as mp

def get_args():
    args_parser = argparse.ArgumentParser(description='Train State Space Encoder Args')  # TODO: make actual args
    args_parser.add_argument('--load-environment-model', action='store_true', default=False,
                             help='flag to continue training on pretrained env_model')
    args_parser.add_argument('--load-policy-model', action='store_true', default=False,
                             help='flag to laod a policy for generating training samples')
    args_parser.add_argument('--save-environment-model-dir', default="trained_models/environment_models/",
                             help='relative path to folder from which a environment model should be loaded.')
    args_parser.add_argument('--env-name', default='MsPacmanNoFrameskip-v0',
                             help='environment to train on (default: MsPacmanNoFrameskip-v0)')
    args_parser.add_argument('--latent-space-model', default='dSSM_DET',
                             help='latent space model (default: dSSM_DET)'
                                  'models = (dSSM_DET, dSSM_VAE, sSSM)')
    args_parser.add_argument('--render', action='store_true', default=False,
                             help='starts an progress that play and render games with the current model')
    args_parser.add_argument('--no-training', action='store_true', default=False,
                             help='true to render a already trained env model')
    args_parser.add_argument('--no-cuda', action='store_true', default=False,
                             help='disables CUDA training')
    args_parser.add_argument('--no-vis', action='store_true', default=False,
                             help='disables visdom visualization')
    args_parser.add_argument('--port', type=int, default=8097,
                             help='port to run the server on (default: 8097)')
    args_parser.add_argument('--save-interval', type=int, default=100,
                             help='save model each n episodes (default: 10)')
    args_parser.add_argument('--skip-frames', type=int, default=1,
                             help='skip frames (default: 1)')
    args_parser.add_argument('--num-episodes', type=int, default=10000000,
                             help='save model each n episodes (default: 10000000)')
    args_parser.add_argument('--batch-size', type=int, default=2,
                             help='batch size (default: 2)')
    args_parser.add_argument('--sample-memory-size', type=int, default=50,
                             help='batch size (default: 50)')
    args_parser.add_argument('--rollout-steps', type=int, default=10,
                             help='train with x rollouts (default: 10)')
    args_parser.add_argument('--lr', type=float, default=0.002,
                             help='learning rate (default: 7e-4)')
    args_parser.add_argument('--weight-decay', type=float, default=0.05,
                             help='weight decay (default: 0)')
    args_parser.add_argument('--reward-prediction-bits', type=int, default=8,
                             help='bits used for reward prediction in reward head of decoder (default: 8)')
    args_parser.add_argument('--env-model-images-save-path', default=None, help='path to save images of rendering')
    args = args_parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    args.grey_scale = False
    args.use_latent_space = True
    return args

def main():
    args = get_args()

    if args.render:
        mp.set_start_method('spawn')

    save_model_path = '{0}{1}_{2}.dat'.format(args.save_environment_model_dir,
                                              args.env_name,
                                              args.latent_space_model)

    env = make_env_ms_pacman(env_id = args.env_name, seed=1, rank=1,
                   log_dir=None, grey_scale=False,
                   skip_frames = args.skip_frames,
                   stack_frames=1)()

    #policy = Policy(obs_shape=env.observation_space.shape, action_space=env.action_space, recurrent_policy=False)

    obs_shape = (env.observation_space.shape[0]*4,)+env.observation_space.shape[1:]
    policy = A2C_PolicyWrapper(AtariModel(obs_shape=obs_shape,action_space=env.action_space.n,use_cuda=args.cuda))

    if args.load_policy_model:
        load_policy_model_path = "trained_models/a2c/"+args.env_name+".pt"
        print("Load policy model", load_policy_model_path)
        saved_policy_state = torch.load(load_policy_model_path, map_location=lambda storage, loc: storage)
        policy.load_state_dict(saved_policy_state)

    model = SSM(model_type=args.latent_space_model,
                observation_input_channels=3,
                state_input_channels=64,
                num_actions=env.action_space.n,
                use_cuda=args.cuda,
                reward_prediction_bits = args.reward_prediction_bits)

    if args.load_environment_model:
        load_environment_model_path = save_model_path
        print("Load environment model", load_environment_model_path)
        saved_state = torch.load(load_environment_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_state)

    if args.cuda:
        policy.cuda()
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)


    if args.render:
        test_process = TestEnvironmentModel(env=copy.deepcopy(env),
                                            environment_model=copy.deepcopy(model),
                                            load_path=save_model_path,
                                            rollout_policy=policy,
                                            args=args)

    if not args.no_training:
        from environment_model.LatentSpaceEncoder.state_space_model_trainer import StateSpaceModelTrainer
        trainer = StateSpaceModelTrainer(args=args, env=env, model=model, policy=policy,
                                         optimizer=optimizer,
                                         save_model_path = save_model_path)
        trainer.train(episoden=args.num_episodes, T=args.rollout_steps)




if __name__ == "__main__":
    main()
