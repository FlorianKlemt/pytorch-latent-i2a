import torch
import torch.nn.functional as F
from environment_model.minipacman_env_model import MiniPacmanEnvModelClassLabels, MiniPacmanEnvModel
from environment_model.env_model_optimizer import EnvironmentModelOptimizer, MiniPacmanEnvironmentModelOptimizer
from rl_visualization.logger import LogTrainEM
from custom_envs import make_custom_env
import random
import gym
import gym_minipacman
from rl_visualization.environment_model.test_environment_model import TestEnvironmentModel

from a2c_models.i2a_mini_model import I2A_MiniModel
from a2c_models.a2c_policy_wrapper import A2C_PolicyWrapper
import argparse
import copy

import multiprocessing as mp

def main():
    args_parser = argparse.ArgumentParser(description='Make Environment Model arguments')
    args_parser.add_argument('--load-environment-model', action='store_true', default=False,
                             help='flag to continue training on pretrained env_model')
    args_parser.add_argument('--save-environment-model-dir', default="trained_models/environment_models/",
                             help='relative path to folder from which a environment model should be loaded.')
    args_parser.add_argument('--load-policy-model', action='store_true', default=False,
                             help='use trained policy model for environment model training')
    args_parser.add_argument('--load-policy-model-dir', default='trained_models/a2c/',
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
    log_path = '{0}env_{1}{2}{3}.log'.format(args.save_environment_model_dir,
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
                                      save_model_path = save_model_path,
                                      log_path = log_path)

    trainer.train_env_model_batchwise(1000000)
    #trainer.train_overfit_on_x_samples(1000000, x_samples=100)
    #trainer.train_env_model(1000)
    #trainer.train_env_model_batchwise_on_rollouts(10000,rollout_length=5)

    if args.render:
        test_process.stop()




class EnvironmentModelTrainer():
    def __init__(self, args, env, policy, optimizer, save_model_path, log_path):
        self.args = args
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.use_cuda = args.cuda
        self.chance_of_random_action = 0.25
        self.batch_size = args.batch_size
        self.save_model_path = save_model_path
        self.sample_memory_size = 100000
        self.log_path = log_path

        if args.vis:
            from visdom import Visdom
            viz = Visdom(port=args.port)
        else:
            viz = None

        self.loss_printer = LogTrainEM(log_name=self.log_path,
                                       batch_size=self.batch_size,
                                       delete_log_file=args.load_environment_model == False,
                                       viz=viz)


    def sample_action_from_distribution(self, actor):
        prob = F.softmax(actor, dim=1)
        action = prob.multinomial(num_samples=1)
        #use_cuda = action.is_cuda

        #if random.random() < self.chance_of_random_action:
        #    action_space = self.env.action_space.n
        #    action_int = random.randint(0, action_space - 1)
        #    action = torch.from_numpy(np.array([action_int])).unsqueeze(0)
        #    if use_cuda:
        #        action = action.cuda()

        return action


    def do_env_step(self, action):
        next_state, reward, done, info = self.env.step(action.item())
        next_state = torch.from_numpy(next_state).unsqueeze(0).float()
        reward = torch.FloatTensor([reward])
        if self.use_cuda:
            next_state = next_state.cuda()
            reward = reward.cuda()
        return  next_state, reward, done, info


    def train_env_model(self, episoden = 10000):
        for i_episode in range(episoden):
            # loss_printer.reset()
            state = self.env.reset()
            state = torch.from_numpy(state).unsqueeze(0).float()
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
                    self.loss_printer.log_loss_and_reward(loss=loss,
                                                          reward_prediction=predicted_reward,
                                                          reward=reward,
                                                          episode=i_episode)

            if i_episode % self.args.save_interval == 0:
                print("Save model", self.save_model_path)
                state_to_save = self.optimizer.model.state_dict()
                torch.save(state_to_save, self.save_model_path)

    def create_x_samples(self, number_of_samples):
        from collections import deque
        sample_memory = deque(maxlen=number_of_samples)

        while len(sample_memory) < number_of_samples:
            state = self.env.reset()
            state = torch.from_numpy(state).unsqueeze(0).float()
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
                state = next_state

                if len(sample_memory) >= number_of_samples:
                    break
        return sample_memory

    def train_env_model_batchwise(self, episoden = 10000):
        from collections import deque
        print("create training data")
        create_n_samples = min(self.batch_size * 2, self.sample_memory_size)
        sample_memory = deque(maxlen=self.sample_memory_size)
        sample_memory.extend(self.create_x_samples(create_n_samples))

        for i_episode in range(episoden):
            # sample a state, next-state pair randomly from replay memory for a training step
            sample_state, sample_action, sample_next_state, sample_reward = [torch.cat(a) for a in zip(*random.sample(sample_memory, self.batch_size))]
            loss, prediction = self.optimizer.optimizer_step(state = sample_state,
                                                             action = sample_action,
                                                             next_state_target = sample_next_state,
                                                             reward_target = sample_reward)
            # log and print infos
            if self.loss_printer:
                (predicted_next_state, predicted_reward) = prediction
                self.loss_printer.log_loss_and_reward(loss=loss,
                                                      reward_prediction=predicted_reward,
                                                      reward=sample_reward,
                                                      episode=i_episode)

            if i_episode % self.args.save_interval==0:
                print("Save model", self.save_model_path)
                state_to_save = self.optimizer.model.state_dict()
                torch.save(state_to_save, self.save_model_path)

            if i_episode != 0 and i_episode % len(sample_memory) == 0:
                print("create more training data")
                sample_memory.extend(self.create_x_samples(create_n_samples))

    def train_overfit_on_x_samples(self, episoden = 10000, x_samples = 100):
        from collections import deque
        sample_memory = deque(maxlen=x_samples)

        self.batch_size = min(self.batch_size, x_samples)

        sample_memory = self.create_x_samples(x_samples)

        from rl_visualization.environment_model.test_environment_model import RenderImaginationCore
        import time
        renderer = RenderImaginationCore(False)

        for i_episode in range(episoden):
            # sample a state, next-state pair randomly from replay memory for a training step
            sample_state, sample_action, sample_next_state, sample_reward = [torch.cat(a) for a in zip(*random.sample(sample_memory, self.batch_size))]
            loss, prediction = self.optimizer.optimizer_step(state = sample_state,
                                                             action = sample_action,
                                                             next_state_target = sample_next_state,
                                                             reward_target = sample_reward)
            # log and print infos
            (predicted_next_state, predicted_reward) = prediction
            if self.loss_printer:
                self.loss_printer.log_loss_and_reward(loss=loss,
                                                      reward_prediction=predicted_reward,
                                                      reward=sample_reward,
                                                      episode=i_episode)

            if i_episode % 100==0:
                print("Save model", self.save_model_path)
                state_to_save = self.optimizer.model.state_dict()
                torch.save(state_to_save, self.save_model_path)

            if i_episode % 500 == 0:
                renderer.render_observation(sample_state[0], sample_state[0], 0, 0, 0)
                renderer.render_observation(sample_next_state[0], predicted_next_state[0], sample_reward.data[0], predicted_reward.data[0], 1)
                time.sleep(1)



    def train_env_model_batchwise_on_rollouts(self, episoden = 10000, rollout_length=5):
        from collections import deque
        sample_memory = deque(maxlen=self.sample_memory_size)   #one sample is a complete rollout here

        for i_episode in range(episoden):
            state = self.env.reset()
            state = torch.from_numpy(state).unsqueeze(0).float()
            if self.use_cuda:
                state = state.cuda()

            done = False
            while not done:
                rollout = []
                for _ in range(rollout_length):
                    # let policy decide on next action and perform it
                    critic, actor = self.policy(state)
                    action = self.sample_action_from_distribution(actor=actor)
                    next_state, reward, done, _ = self.do_env_step(action=action)
                    rollout.append([state, action, next_state, reward])
                    state = next_state

                # add current rollout to replay memory
                sample_memory.append(rollout)

                # sample a rollout randomly from replay memory for a training step
                if len(sample_memory) > self.batch_size:
                    sample_rollouts = random.sample(sample_memory, self.batch_size) #[[state, action, next_state, reward],...],[...]
                    loss, prediction = self.optimizer.rollout_optimizer_step(sample_rollouts)

                    # log and print infos
                    if self.loss_printer:
                        (_, predicted_reward) = prediction
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
