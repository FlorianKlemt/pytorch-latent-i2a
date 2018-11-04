import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr | i2a')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--distill-coef', type=float, default=10,
                        help='distill term coefficient only used for i2a (default: 10)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--num-stack', type=int, default=4,
                        help='number of frames to stack (default: 4)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--render-game',  action='store_true', default=False,
                        help='starts an progress that play and render games with the current model')
    parser.add_argument('--num-frames', type=int, default=100e6,
                        help='number of frames to train (default: 100e6)')
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)'
                             'I2A envs examples: '
                             'RegularMiniPacmanNoFrameskip-v0, '
                             'HuntMiniPacmanNoFrameskip-v0, '
                             'MsPacmanNoFrameskip-v0)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--load-model',  action='store_true', default=False,
                        help='load existing model')
    parser.add_argument('--load-environment-model-dir', default="trained_models/environment_models/",
                        help='relative path to folder from which a environment model should be loaded.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--no-vis', action='store_true', default=False,
                        help='disables visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    parser.add_argument('--grey_scale', action='store_true', default=False,
                             help='True to convert to grey_scale images')
    parser.add_argument('--train-on-200x160-pixel', action='store_true', default=False,
                        help='True to use 200x160 image, False for downsampled image')
    parser.add_argument('--i2a-rollout-steps', type=int, default=2,
                        help='number of steps the imagination core rollouts in the I2A training (default: 5)')
    parser.add_argument('--no-training', action='store_true', default=False,
                        help='true to render an already trained model')
    parser.add_argument('--environment-model', default='dSSM_DET',
                        help='environment model (default: dSSM_DET)'
                             'mini pacman models = (MiniModel, MiniModelLabels, CopyModel)'
                             'latent space models = (dSSM_DET, dSSM_VAE, sSSM)')
    parser.add_argument('--reward-prediction-bits', type=int, default=8,
                             help='Only used when train with latent space model'
                                  'bits used for reward prediction in reward head of decoder (default: 8)')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis
    if args.no_training:
        args.num_frames = 0
        args.render_game = True
        args.load_model = True

    args.save_environment_model_dir = args.load_environment_model_dir
    args.load_environment_model = True
    return args
