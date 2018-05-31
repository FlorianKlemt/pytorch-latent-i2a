import torch
from torch.autograd import Variable
import torch.nn.functional as F
import gym
import gym_minipacman
from I2A.EnvironmentModel.MiniPacmanEnvModel import MiniPacmanEnvModelClassLabels, MiniPacmanEnvModel
from I2A.EnvironmentModel.EnvironmentModelOptimizer import EnvironmentModelOptimizer, MiniPacmanEnvironmentModelOptimizer
from I2A.EnvironmentModel.RenderTrainEM import RenderTrainEM
from logger import LogTrainEM
from custom_envs import make_custom_env
import random

from I2A.EnvironmentModel.test_environment_model import TestEnvironmentModel

from A2C_Models.I2A_MiniModel import I2A_MiniModel
from A2C_Models.A2C_PolicyWrapper import A2C_PolicyWrapper
import argparse
import numpy as np
import os
import copy

import multiprocessing as mp

def main():
    args_parser = argparse.ArgumentParser(description='Make Environment Model arguments')
    args_parser.add_argument('--load-environment-model', action='store_true', default=False,
                             help='flag to continue training on pretrained env_model')
    args_parser.add_argument('--save-environment-model-dir', default="../../trained_models/environment_models/",
                             help='relative path to folder from which a environment model should be loaded.')
    args_parser.add_argument('--load-policy-model', action='store_true', default=False,
                             help='use trained policy model for environment model training')
    args_parser.add_argument('--load-policy-model-dir', default='../../trained_models/a2c/',
                             help='directory to save agent logs (default: ./trained_models/)')
    args_parser.add_argument('--env-name', default='RegularMiniPacmanNoFrameskip-v0',
                             help='environment to train on (default: RegularMiniPacmanNoFrameskip-v0)')
    args_parser.add_argument('--render',  action='store_true', default=False,
                             help='starts an progress that play and render games with the current model')
    args_parser.add_argument('--no-cuda', action='store_true', default=False,
                             help='disables CUDA training')
    args_parser.add_argument('--no-vis', action='store_true', default=False,
                             help='disables visdom visualization')
    args_parser.add_argument('--port', type=int, default=8097,
                             help='port to run the server on (default: 8097)')
    args_parser.add_argument('--grey_scale', action='store_true', default=False,
                             help='True to convert to grey_scale images')
    args_parser.add_argument('--save-interval', type=int, default=10,
                             help='save model each n episodes (default: 10)')
    args_parser.add_argument('--use-class-labels', action='store_true', default=False,
                             help='true to use pixelwise cross-entropy-loss and make'
                                  'the color of each pixel a classification task')
    args_parser.add_argument('--batch-size', type=int, default=100,
                             help='batch size (default: 100)')
    args_parser.add_argument('--lr', type=float, default=7e-4, #1e-4
                             help='learning rate (default: 7e-4)')
    args_parser.add_argument('--eps', type=float, default=1e-5, #1e-8
                             help='RMSprop optimizer epsilon (default: 1e-5)')
    args_parser.add_argument('--weight-decay', type=float, default=0,
                             help='weight decay (default: 0)')
    args_parser.add_argument('--reward-loss-coef', type=float, default=0.5,
                             help='reward loss coef (default: 0.5)')
    args = args_parser.parse_args()

    #args.save_environment_model_dir = os.path.join('../../', 'trained_models/environment_models/')
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    if args.render:
        mp.set_start_method('spawn')

    color_prefix = 'grey_scale' if args.grey_scale else 'RGB'
    class_labels_prefix = '_labels' if args.use_class_labels else ''
    save_model_path = '{0}{1}{2}{3}.dat'.format(args.save_environment_model_dir,
                                                args.env_name,
                                                color_prefix,
                                                class_labels_prefix)

    load_policy_model_path = '{0}{1}.pt'.format(args.load_policy_model_dir, args.env_name)

    env = make_custom_env(args.env_name, seed=1, rank=1, log_dir=None, grey_scale=args.grey_scale)() #wtf

    policy = build_policy(env=env,
                          load_policy_model=args.load_policy_model,
                          load_policy_model_path=load_policy_model_path,
                          use_cuda=args.cuda)

    #relative_load_environment_model_dir = os.path.join('../../', args.load_environment_model_dir)
    environment_model = build_em_model(env=env,
                                       load_environment_model=args.load_environment_model,
                                       load_environment_model_path=save_model_path,
                                       use_cuda=args.cuda,
                                       use_class_labels=args.use_class_labels)

    if args.use_class_labels:
        optimizer = MiniPacmanEnvironmentModelOptimizer(model=environment_model, args=args)
    else:
        optimizer = EnvironmentModelOptimizer(model=environment_model, args=args)

    if args.render:
        test_process = TestEnvironmentModel(env = env,
                                            environment_model=copy.deepcopy(environment_model),
                                            load_path=save_model_path,
                                            rollout_policy=policy,
                                            args=args)


    trainer = EnvironmentModelTrainer(args = args,
                                      env=env,
                                      policy=policy,
                                      optimizer=optimizer,
                                      save_model_path = save_model_path)

    trainer.train_env_model_batchwise(1000000)
    #trainer.train_env_model(1000)

    if args.render:
        test_process.stop()




class EnvironmentModelTrainer():
    def __init__(self, args, env, policy, optimizer, save_model_path):
        self.args = args
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.use_cuda = args.cuda
        self.chance_of_random_action = 0.25
        self.batch_size = args.batch_size
        self.save_model_path = save_model_path
        self.sample_memory_size = 10000

        if args.vis:
            from visdom import Visdom
            viz = Visdom(port=args.port)
        else:
            viz = None

        self.loss_printer = LogTrainEM(log_name="em_trainer_" + args.env_name + ".log",
                                       delete_log_file=args.load_environment_model == False,
                                       viz=viz)


    def sample_action_from_distribution(self, actor):
        prob = F.softmax(actor, dim=1)
        action = prob.multinomial().data
        use_cuda = action.is_cuda

        if random.random() < self.chance_of_random_action:
            action_space = self.env.action_space.n
            action_int = random.randint(0, action_space - 1)
            action = torch.from_numpy(np.array([action_int])).unsqueeze(0)
            if use_cuda:
                action = action.cuda()

        action = Variable(action)
        return action


    def do_env_step(self, action):
        next_state, reward, done, info = self.env.step(action.data[0][0])
        next_state = Variable(torch.from_numpy(next_state).unsqueeze(0)).float()
        reward = Variable(torch.from_numpy(np.array([reward]))).float()
        if self.use_cuda:
            next_state = next_state.cuda()
            reward = reward.cuda()
        return  next_state, reward, done, info


    def train_env_model(self, episoden = 10000):
        for i_episode in range(episoden):
            # loss_printer.reset()
            state = self.env.reset()
            state = Variable(torch.from_numpy(state).unsqueeze(0)).float()
            if self.use_cuda:
                state = state.cuda()

            done = False
            while not done:
                # let policy decide on next action and perform it
                critic, actor = self.policy(state)
                action = self.sample_action_from_distribution(actor=actor)

                next_state, reward, done, _ = self.do_env_step(action=action)

                loss, prediction = self.optimizer.optimizer_step(state=state,
                                                                 action=action,
                                                                 next_state_target=next_state,
                                                                 reward_target=reward)
                state = next_state


                # log and print infos
                if self.loss_printer:
                    (predicted_next_state, predicted_reward) = prediction
                    self.loss_printer.log_loss_and_reward(loss, predicted_reward, reward)
                    if self.loss_printer.frames % 100 == 0:
                        self.loss_printer.print_episode(episode=i_episode)

            if i_episode % self.args.save_interval == 0:
                print("Save model", self.save_model_path)
                state_to_save = self.optimizer.model.state_dict()
                torch.save(state_to_save, self.save_model_path)

    def train_env_model_batchwise(self, episoden = 10000):
        from collections import deque
        sample_memory = deque(maxlen=self.sample_memory_size)

        for i_episode in range(episoden):
            state = self.env.reset()
            state = Variable(torch.from_numpy(state).unsqueeze(0)).float()
            if self.use_cuda:
                state = state.cuda()

            done = False
            while not done:
                # let policy decide on next action and perform it
                critic, actor = self.policy(state)
                action = self.sample_action_from_distribution(actor=actor)
                next_state, reward, done, _ = self.do_env_step(action=action)

                # add current state, next-state pair to replay memory
                sample_memory.append([state, action, next_state, reward])

                # sample a state, next-state pair randomly from replay memory for a training step
                if len(sample_memory) > self.batch_size:
                    sample_state, sample_action, sample_next_state, sample_reward = [torch.cat(a) for a in zip(*random.sample(sample_memory, self.batch_size))]
                    loss, prediction = self.optimizer.optimizer_step(state = sample_state,
                                                                     action = sample_action,
                                                                     next_state_target = sample_next_state,
                                                                     reward_target = sample_reward)

                state = next_state

                # log and print infos
                if self.loss_printer:
                    (predicted_next_state, predicted_reward) = prediction
                    self.loss_printer.log_loss_and_reward(loss, predicted_reward, reward)
                    if self.loss_printer.frames % 10 == 0:
                        self.loss_printer.print_episode(episode=i_episode)

            if i_episode % self.args.save_interval==0:
                print("Save model", self.save_model_path)
                state_to_save = self.optimizer.model.state_dict()
                torch.save(state_to_save, self.save_model_path)


def build_policy(env, load_policy_model, load_policy_model_path, use_cuda):
    policy = A2C_PolicyWrapper(I2A_MiniModel(obs_shape=env.observation_space.shape, action_space=env.action_space.n, use_cuda=use_cuda))
    if use_cuda:
        policy.cuda()

    if load_policy_model:
        saved_state = torch.load(load_policy_model_path, map_location=lambda storage, loc: storage)
        print("Load Policy Model", load_policy_model_path)
        policy.load_state_dict(saved_state)

    return policy

def build_em_model(env, load_environment_model=False, load_environment_model_path=None, use_cuda=True, use_class_labels=False):
    #TODO: @future self: once we have latent space models change the next line
    if use_class_labels:
        EMModel = MiniPacmanEnvModelClassLabels
        labels = 7
        em_obs_shape = (labels, env.observation_space.shape[1], env.observation_space.shape[2])
    else:
        EMModel = MiniPacmanEnvModel
        em_obs_shape = env.observation_space.shape

    reward_bins = env.unwrapped.reward_bins #[0., 1., 2., 5., 0.] for regular

    environment_model = EMModel(obs_shape=em_obs_shape, #env.observation_space.shape,  # 4
                                num_actions=env.action_space.n,
                                reward_bins=reward_bins,
                                use_cuda=use_cuda)

    if load_environment_model:
        print("Load environment model", load_environment_model_path)
        saved_state = torch.load(load_environment_model_path, map_location=lambda storage, loc: storage)
        environment_model.load_state_dict(saved_state)

    if use_cuda:
        environment_model.cuda()

    return environment_model

if __name__ == '__main__':
    main()