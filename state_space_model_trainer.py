import torch
import random
import torch.functional as F
import gym
from LatentSpaceEncoder.models_from_paper.dSSM import dSSM_DET
from LatentSpaceEncoder.models_from_paper.sSSM import sSSM
import argparse
from bigger_models import Policy
from a2c_models.a2c_policy_wrapper import A2C_PolicyWrapper

from LatentSpaceEncoder.env_encoder import make_env
from custom_envs import ClipAtariFrameSizeTo200x160
from rl_visualization.logger import LogTrainEM
from rl_visualization.environment_model.test_environment_model import TestEnvironmentModel
import copy
import multiprocessing as mp
#from envs import make_env

def main():
    args_parser = argparse.ArgumentParser(description='StateSpaceEncoder')  #TODO: make actual args
    args = args_parser.parse_args()
    args.use_cuda = True
    args.cuda = args.use_cuda
    args.batch_size = 5
    args.save_env_model_dir = "trained_models/environment_models/"
    args.vis = True
    args.port = 8097
    args.save_interval = 20
    args.render = True
    args.env_name ="MsPacmanNoFrameskip-v0"
    args.grey_scale = False
    args.load_environment_model = False

    if args.render:
        mp.set_start_method('spawn')

    save_model_path = '{0}{1}{2}.dat'.format(args.save_env_model_dir,
                                                  args.env_name,
                                                  "state_space")

    #env = make_env("MsPacmanNoFrameskip-v0", 1, 1, None, False)()
    env = make_env(args.env_name, 1, 1, None, False, False)()
    env = ClipAtariFrameSizeTo200x160(env=env)

    policy = Policy(obs_shape=env.observation_space.shape, action_space=env.action_space, recurrent_policy=False)
    model = dSSM_DET(observation_input_channels=3, state_input_channels=64, num_actions=env.action_space.n, use_cuda=True)
    #model = sSSM(observation_input_channels=3, state_input_channels=64, num_actions=env.action_space.n, use_cuda=True)
    if args.load_environment_model:
        load_environment_model_path = save_model_path
        print("Load environment model", load_environment_model_path)
        saved_state = torch.load(load_environment_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_state)

    if args.use_cuda:
        policy.cuda()
        model.cuda()
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0)  #0.00005, 1e-5
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_criterion = torch.nn.MSELoss()

    if args.render:
        test_process = TestEnvironmentModel(env=env,
                                            environment_model=copy.deepcopy(model),
                                            load_path=save_model_path,
                                            rollout_policy=policy,
                                            args=args)

    trainer = StateSpaceModelTrainer(args=args, env=env, model=model, policy=policy, optimizer=optimizer, loss_criterion=loss_criterion,
                                     save_model_path = save_model_path)
    #import time
    #time.sleep(100000000)
    #trainer.train_env_model_batchwise(episoden=100000)

    trainer.train_new(episoden=1000000)


class StateSpaceModelTrainer():
    def __init__(self, args, env, model, policy, optimizer, loss_criterion, save_model_path):
        self.model = model
        self.args = args
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.loss_criterion = loss_criterion
        self.use_cuda = args.use_cuda
        self.batch_size = args.batch_size
        self.sample_memory_size = 200 #500 #100000
        self.frame_stack = 1

        self.log_path = '{0}env_{1}{2}.log'.format(args.save_env_model_dir,
                                                   args.env_name,
                                                   "state_space")

        self.save_model_path = save_model_path



        if args.vis:
            from visdom import Visdom
            viz = Visdom(port=args.port)
        else:
            viz = None
        self.loss_printer = LogTrainEM(log_name=self.log_path,
                                       batch_size=self.batch_size,
                                       delete_log_file=True,
                                       viz=viz)


    def train_new(self, episoden = 1000, T=10):
        from collections import deque
        print("create training data")
        create_n_samples = min(self.batch_size * 2, self.sample_memory_size)
        sample_memory = deque(maxlen=self.sample_memory_size)
        sample_memory.extend(self.create_x_samples_T_forward(create_n_samples, T))
        import torch.nn as nn
        #criterion = nn.KLDivLoss()
        criterion = torch.nn.BCELoss()

        for i_episode in range(episoden):
            # sample a state, next-state pair randomly from replay memory for a training step
            sample_observation, sample_action_T, sample_next_observation_T, sample_reward_T = [torch.cat(a) for a in zip
                (*random.sample(sample_memory, self.batch_size))]

            sample_next_observation_T = torch.clamp(sample_next_observation_T, 0.00000001, 1)

            image_log_probs = self.model.forward_multiple(sample_observation, sample_action_T)
            #image_log_probs, _ = self.model(sample_observation, sample_action)

            image_log_probs = torch.clamp(image_log_probs, 0.00000001, 1)
            pre_log = image_log_probs
            image_log_probs = torch.log(image_log_probs)

            predicted_bernoulli = torch.distributions.bernoulli.Bernoulli(logits=image_log_probs)

            ground_truth_bernoulli = torch.distributions.bernoulli.Bernoulli(logits=torch.log(sample_next_observation_T))
            #loss = torch.distributions.kl.kl_divergence(predicted_bernoulli, ground_truth_bernoulli)
            kl_loss = torch.distributions.kl._kl_bernoulli_bernoulli(predicted_bernoulli, ground_truth_bernoulli)
            kl_loss = torch.mean(kl_loss)

            reconstruction_loss = criterion(pre_log, sample_next_observation_T)

            loss = reconstruction_loss + kl_loss

            #loss = kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log and print infos
            if self.loss_printer:
                self.loss_printer.log_loss_and_reward((loss, loss), torch.zeros(1), torch.zeros(1), i_episode)
                if self.loss_printer.frames % 10 == 0:
                    self.loss_printer.print_episode(episode=i_episode)

            if i_episode % self.args.save_interval == 0:
                print("Save model", self.save_model_path)
                state_to_save = self.model.state_dict()
                torch.save(state_to_save, self.save_model_path)

            if i_episode != 0 and i_episode % create_n_samples == 0:
                print("create more training data ", len(sample_memory))
                sample_memory.extend(self.create_x_samples_T_forward(create_n_samples, T))




    def train_env_model_batchwise(self, episoden = 10000):
        from collections import deque
        print("create training data")
        create_n_samples = min(self.batch_size * 2, self.sample_memory_size)
        sample_memory = deque(maxlen=self.sample_memory_size)
        sample_memory.extend(self.create_x_samples(create_n_samples))

        for i_episode in range(episoden):
            # sample a state, next-state pair randomly from replay memory for a training step
            sample_observation, sample_action, sample_next_observation, sample_reward = [torch.cat(a) for a in zip
                (*random.sample(sample_memory, self.batch_size))]

            image_log_probs, reward_log_probs = self.model(sample_observation, sample_action)
            loss = self.loss_criterion(image_log_probs, sample_next_observation)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            # log and print infos
            if self.loss_printer:
                self.loss_printer.log_loss_and_reward((loss,loss), torch.zeros(1), torch.zeros(1), i_episode)
                if self.loss_printer.frames % 10 == 0:
                    self.loss_printer.print_episode(episode=i_episode)

            if i_episode % self.args.save_interval == 0:
                print("Save model", self.save_model_path)
                state_to_save = self.model.state_dict()
                torch.save(state_to_save, self.save_model_path)

            #if i_episode != 0 and i_episode % len(sample_memory) == 0:
            if i_episode != 0 and i_episode % create_n_samples == 0:
                print("create more training data ", len(sample_memory))
                sample_memory.extend(self.create_x_samples(create_n_samples))




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
                state_stack = None
                action_stack = None
                for _ in range(self.frame_stack):
                    # let policy decide on next action and perform it
                    value, action, _, _ = self.policy.act(inputs=state, states=None, masks=None)  #no state and mask
                    if state_stack is not None and action_stack is not None:
                        state_stack = torch.cat((state_stack,state))
                        action_stack = torch.cat((action_stack, action))
                    else:
                        state_stack = state
                        action_stack = action

                    state, reward, done, _ = self.do_env_step(action=action)


                # add current state, next-state pair to replay memory
                #sample_memory.append([state, action, next_state, reward])
                #state = next_state
                target_state = torch.cat((state_stack[1:], state))
                sample_memory.append([state_stack, action_stack, target_state, reward])

                if len(sample_memory) >= number_of_samples:
                    break
        return sample_memory



    def create_x_samples_T_forward(self, number_of_samples, T):
        from collections import deque
        sample_memory = deque(maxlen=number_of_samples)

        while len(sample_memory) < number_of_samples:
            state = self.env.reset()
            state = torch.from_numpy(state).unsqueeze(0).float()
            if self.use_cuda:
                state = state.cuda()
            done = False
            while not done:
                state_stack = None
                action_stack = None
                reward_stack = None
                for _ in range(T):
                    # let policy decide on next action and perform it
                    value, action, _, _ = self.policy.act(inputs=state, states=None, masks=None)  #no state and mask
                    if state_stack is not None and action_stack is not None:
                        state_stack = torch.cat((state_stack,state))
                        action_stack = torch.cat((action_stack, action))
                    else:
                        state_stack = state
                        action_stack = action

                    state, reward, done, _ = self.do_env_step(action=action)
                    if reward_stack is not None:
                        reward_stack = torch.cat((reward_stack, reward))
                    else:
                        reward_stack = reward

                target_state = torch.cat((state_stack[1:], state))
                #unsqueeze action, target_state and reward stack for batch dimension
                sample_memory.append([state_stack[:1], action_stack.unsqueeze(0), target_state.unsqueeze(0), reward_stack.unsqueeze(0)])

                if len(sample_memory) >= number_of_samples:
                    break
        return sample_memory


    def sample_action_from_distribution(self, actor):
        prob = F.softmax(actor, dim=1)
        action = prob.multinomial(num_samples=1)
        return action


    def do_env_step(self, action):
        next_state, reward, done, info = self.env.step(action.item())
        next_state = torch.from_numpy(next_state).unsqueeze(0).float()
        reward = torch.FloatTensor([reward])
        if self.use_cuda:
            next_state = next_state.cuda()
            reward = reward.cuda()
        return  next_state, reward, done, info


if __name__ == "__main__":
    main()