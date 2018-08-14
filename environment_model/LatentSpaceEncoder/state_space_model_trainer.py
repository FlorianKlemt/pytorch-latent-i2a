import torch
import random
import torch.functional as F

from rl_visualization.logger import LogTrainEM

from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
import math


class StateSpaceModelTrainer():
    def __init__(self, args, env, model, policy, optimizer, save_model_path):
        self.model = model
        self.args = args
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.use_cuda = args.cuda
        self.sample_memory_on_gpu = False
        self.batch_size = args.batch_size
        self.sample_memory_size = args.sample_memory_size
        self.log_path = '{0}{1}_{2}.log'.format(args.save_environment_model_dir,
                                                args.env_name,
                                                args.latent_space_model)

        self.save_model_path = save_model_path

        self.frame_criterion = torch.nn.BCELoss()
        self.reward_criterion = torch.nn.BCEWithLogitsLoss()

        if args.latent_space_model == "dSSM_DET":
            self.train_episode = self.train_episode_dSSM
        else:
            self.train_episode = self.train_episode_sSSM

        if args.vis:
            from visdom import Visdom
            viz = Visdom(port=args.port)
        else:
            viz = None
        self.loss_printer = LogTrainEM(log_name=self.log_path,
                                       batch_size=self.batch_size,
                                       delete_log_file=True,
                                       viz=viz)


    def numerical_reward_to_bit_array(self, rewards):
        reward_prediction_bits = self.args.reward_prediction_bits
        # one bit for sign, and one bit for 0
        reward_prediction_numerical_bits = reward_prediction_bits - 2
        max_representable_reward = int(math.pow(2, reward_prediction_numerical_bits) - 1)
        if self.args.cuda:
            r_true = torch.cuda.FloatTensor(rewards.shape[0], rewards.shape[1], reward_prediction_bits).fill_(0)
        else:
            r_true = torch.FloatTensor(rewards.shape[0], rewards.shape[1], reward_prediction_bits).fill_(0)
        for i in range(rewards.shape[0]):
            for j in range(rewards.shape[1]):
                true_reward = math.floor(rewards[i, j].item())  # they floor in the paper too
                if true_reward < -max_representable_reward:
                    print("True Reward too small to represent: ", true_reward, "<", -max_representable_reward)
                    true_reward = -max_representable_reward
                if true_reward > max_representable_reward:
                    print("True Reward too large to represent: ", true_reward, ">", max_representable_reward)
                    true_reward = max_representable_reward

                r_true[i, j, 0] = int(true_reward == 0)
                r_true[i, j, 1] = int(true_reward < 0)
                number_str_format = '{0:0'+str(reward_prediction_numerical_bits)+'b}'
                bits = [int(x) for x in list(number_str_format.format(abs(true_reward)))]
                for n in range(2, reward_prediction_bits):
                    r_true[i, j, n] = bits[n - 2]
                # print(r_true[i,j], true_reward)
        return r_true

    def train_episode_dSSM(self, sample):
        sample_observation_initial_context, sample_action_T, sample_next_observation_T, sample_reward_T = sample
        image_probs, reward_probs = self.model.forward_multiple(sample_observation_initial_context, sample_action_T)

        # reward loss
        true_reward = self.numerical_reward_to_bit_array(sample_reward_T)
        reward_loss = self.reward_criterion(reward_probs, true_reward)

        # image loss
        reconstruction_loss = self.frame_criterion(image_probs, sample_next_observation_T)

        loss = reconstruction_loss + 1e-2 * reward_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # log and print infos
        if self.loss_printer:
            # The minimal cross entropy between the distributions p and q is the entropy of p
            # so if they are equal the loss is equal to the distribution of p
            true_entropy = Bernoulli(probs=sample_next_observation_T).entropy()
            entropy_normalized_loss = reconstruction_loss - true_entropy.mean()
            return entropy_normalized_loss, reward_loss
        return reconstruction_loss, reward_loss

    def train_episode_sSSM(self, sample):
        sample_observation_initial_context, sample_action_T, sample_next_observation_T, sample_reward_T = sample
        # sample_next_observation_T = torch.clamp(sample_next_observation_T, 0.001, 1)

        image_probs, reward_probs, \
        (total_z_mu_prior, total_z_sigma_prior, total_z_mu_posterior, total_z_sigma_posterior) \
            = self.model.forward_multiple(sample_observation_initial_context, sample_action_T)

        true_reward = self.numerical_reward_to_bit_array(sample_reward_T)
        reward_loss = self.reward_criterion(reward_probs, true_reward)
        # print(reward_loss)

        reconstruction_loss = self.frame_criterion(image_probs, sample_next_observation_T)
        # print("Rec loss: ", reconstruction_loss)

        prior_gaussian = Normal(loc=total_z_mu_prior, scale=total_z_sigma_prior)
        posterior_gaussian = Normal(loc=total_z_mu_posterior, scale=total_z_sigma_posterior)
        kl_div_loss = torch.distributions.kl.kl_divergence(prior_gaussian, posterior_gaussian)
        frame_loss = reconstruction_loss + kl_div_loss.mean()  # loss is elbo

        loss = frame_loss + 1e-2 * reward_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # log and print infos
        if self.loss_printer:
            # The minimal cross entropy between the distributions p and q is the entropy of p
            # so if they are equal the loss is equal to the distribution of p
            true_entropy = Bernoulli(probs=sample_next_observation_T).entropy()
            entropy_normalized_loss = reconstruction_loss - true_entropy.mean()
            return entropy_normalized_loss + kl_div_loss.mean(), reward_loss
        return frame_loss, reward_loss


    def train(self, episoden = 1000, T=10, initial_context_size=3, policy_frame_stack=4):
        from collections import deque
        print("create training data")
        create_n_samples = min(self.batch_size * 5, self.sample_memory_size)
        sample_memory = deque(maxlen=self.sample_memory_size)
        sample_memory.extend(self.create_x_samples_T_steps(create_n_samples, T, initial_context_size=initial_context_size, policy_frame_stack=policy_frame_stack))

        for i_episode in range(episoden):
            # sample a state, next-state pair randomly from replay memory for a training step
            sample = [torch.cat(a) for a in zip(*random.sample(sample_memory, self.batch_size))]
            if self.use_cuda:
                sample = [s.cuda() for s in sample]

            loss, reward_loss = self.train_episode(sample=sample)

            # log and print infos
            if self.loss_printer:
                self.loss_printer.log_loss_and_reward((loss, reward_loss), torch.zeros(1), torch.zeros(1), i_episode)
                if self.loss_printer.frames % 10 == 0:
                    self.loss_printer.print_episode(episode=i_episode)

            if i_episode % self.args.save_interval == 0:
                print("Save model", self.save_model_path)
                state_to_save = self.model.state_dict()
                torch.save(state_to_save, self.save_model_path)

            if i_episode != 0 and i_episode % create_n_samples == 0:
                print("create more training data ", len(sample_memory))
                sample_memory.extend(self.create_x_samples_T_steps(create_n_samples, T,
                                                                   initial_context_size=initial_context_size,
                                                                   policy_frame_stack=policy_frame_stack))




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
            if self.sample_memory_on_gpu:
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



    def create_x_samples_T_steps(self, number_of_samples, T, initial_context_size, policy_frame_stack):
        from collections import deque
        sample_memory = deque(maxlen=number_of_samples)

        while len(sample_memory) < number_of_samples:
            done = False
            state = self.env.reset()
            state = torch.from_numpy(state).unsqueeze(0).float()

            frame_stack = state.repeat(1,policy_frame_stack,1,1)

            if self.sample_memory_on_gpu:
                frame_stack = frame_stack.cuda()

            from random import randint
            for i in range(randint(1, 100)):
                stack = frame_stack.cuda()
                value, action, _, _ = self.policy.act(inputs=stack, states=None, masks=None)  # no state and mask
                state, reward, done, _ = self.do_env_step(action=action)
                frame_stack = torch.cat((frame_stack[:, 3:], state), dim=1)

            while not done:
                initial_context_stack = None
                action_stack = None
                reward_stack = None
                target_state_stack = None

                for i in range(initial_context_size):
                    value, action, _, _ = self.policy.act(inputs=frame_stack.cuda(), states=None, masks=None)  # no state and mask
                    state, reward, done, _ = self.do_env_step(action=action)
                    if initial_context_stack is not None:
                        initial_context_stack = torch.cat((initial_context_stack, state))
                    else:
                        initial_context_stack = state
                    frame_stack = torch.cat((frame_stack[:, 3:], state), dim=1)


                for i in range(T):
                    # let policy decide on next action and perform it
                    value, action, _, _ = self.policy.act(inputs=frame_stack.cuda(), states=None, masks=None)  #no state and mask
                    state, reward, done, _ = self.do_env_step(action=action)
                    if target_state_stack is not None:
                        target_state_stack = torch.cat((target_state_stack,state))
                    else:
                        target_state_stack = state

                    if action_stack is not None:
                        action_stack = torch.cat((action_stack, action))
                    else:
                        action_stack = action

                    if reward_stack is not None:
                        reward_stack = torch.cat((reward_stack, reward))
                    else:
                        reward_stack = reward

                    frame_stack = torch.cat((frame_stack[:, 3:], state), dim=1)

                #unsqueeze initial_context, action, target_state and reward stack for batch dimension
                sample_memory.append([initial_context_stack.unsqueeze(0), action_stack.unsqueeze(0),
                                      target_state_stack.unsqueeze(0), reward_stack.unsqueeze(0)])


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
        if self.sample_memory_on_gpu:
            next_state = next_state.cuda()
            reward = reward.cuda()
        return  next_state, reward, done, info