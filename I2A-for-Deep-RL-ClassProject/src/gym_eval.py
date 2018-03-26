from __future__ import division
import argparse
import torch
from environment import atari_env
from utils import read_config, setup_logger
from ModelAlias import Model
from player_util import Agent
import gym
import logging
from gym.configuration import undo_logger_setup
import os

os.environ["OMP_NUM_THREADS"] = "1"

undo_logger_setup()

parser = argparse.ArgumentParser(description='Model_EVAL')
parser.add_argument(
    '--env',
    default='Pong-v0',
    metavar='ENV',
    help='environment to train on (default: Pong-v0)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--num-episodes',
    type=int,
    default=100,
    metavar='NE',
    help='how many episodes in evaluation (default: 100)')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--log-dir',
    default='logs/',
    metavar='LG',
    help='folder to save logs')
parser.add_argument(
    '--render',
    default=False,
    metavar='R',
    help='Watch game as it being played')
parser.add_argument(
    '--render-freq',
    type=int,
    default=1,
    metavar='RF',
    help='Frequency to watch rendered game play')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=100000,
    metavar='M',
    help='maximum length of an episode (default: 100000)')
parser.add_argument(
    '--gpu-id',
    type=int,
    default=-1,
    help='GPU to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--skip_rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')
args = parser.parse_args()

setup_json = read_config(args.env_config)
env_conf = setup_json["Default"]
for i in setup_json.keys():
    if i in args.env:
        env_conf = setup_json[i]

gpu_id = args.gpu_id

torch.manual_seed(args.seed)
if gpu_id >= 0:
    torch.cuda.manual_seed(args.seed)

saved_state = torch.load(
    '{0}{1}.dat'.format(args.load_model_dir, args.env),
    map_location=lambda storage, loc: storage)

log = {}
setup_logger('{}_mon_log'.format(args.env), r'{0}{1}_mon_log'.format(
    args.log_dir, args.env))
log['{}_mon_log'.format(args.env)] = logging.getLogger(
    '{}_mon_log'.format(args.env))

d_args = vars(args)
for k in d_args.keys():
    log['{}_mon_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

env = atari_env("{}".format(args.env), env_conf, args)
num_tests = 0
reward_total_sum = 0
player = Agent(None, env, args, None)
player.model = Model(player.env.observation_space.shape[0],
                     player.env.action_space.n, gpu_id >= 0)

if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model = player.model.cuda()

player.env = gym.wrappers.Monitor(
    player.env, "{}_monitor".format(args.env), force=True)
player.model.eval()

for i_episode in range(args.num_episodes):
    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
    player.eps_len += 2
    reward_sum = 0
    while True:
        if args.render:
            if i_episode % args.render_freq == 0:
                player.env.render()

        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())

        player.action_test()
        reward_sum += player.reward

        if player.done and player.info['ale.lives'] > 0 and not player.max_length:  # ugly hack need to clean this up
            state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        elif player.done or player.max_length:
            player.current_life = 0
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_mon_log'.format(args.env)].info(
                "reward sum: {0}, reward mean: {1:.4f}".format(
                    reward_sum, reward_mean))
            player.eps_len = 0

            break
