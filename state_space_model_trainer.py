import torch
import random
import torch.functional as F
import gym
from LatentSpaceEncoder.models_from_paper.dSSM import dSSM_DET
from LatentSpaceEncoder.models_from_paper.sSSM import sSSM
from LatentSpaceEncoder.models_from_paper.dSSM_VAE import dSSM_VAE
from LatentSpaceEncoder.models_from_paper.state_space_model import SSM

import argparse
from bigger_models import Policy
from a2c_models.a2c_policy_wrapper import A2C_PolicyWrapper
from a2c_models.atari_model import AtariModel

from LatentSpaceEncoder.env_encoder import make_env
from custom_envs import ClipAtariFrameSizeTo200x160
from rl_visualization.logger import LogTrainEM
from rl_visualization.environment_model.test_environment_model import TestEnvironmentModel
import copy
import multiprocessing as mp
#from envs import make_env

from torch.distributions.normal import Normal

def main():
    args_parser = argparse.ArgumentParser(description='Train State Space Encoder Args')  #TODO: make actual args
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
    args_parser.add_argument('--lr', type=float, default=0.002,
                             help='learning rate (default: 7e-4)')
    args_parser.add_argument('--weight-decay', type=float, default=0.05,
                             help='weight decay (default: 0)')
    args = args_parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    args.save_interval = 20
    args.grey_scale = False
    args.use_latent_space = True

    if args.render:
        mp.set_start_method('spawn')

    save_model_path = '{0}{1}_{2}.dat'.format(args.save_environment_model_dir,
                                              args.env_name,
                                              args.latent_space_model)

    env = make_env(env_id = args.env_name, seed=1, rank=1,
                   log_dir=None, grey_scale=False,
                   skip_frames = args.skip_frames,
                   stack_frames=1)()
    env = ClipAtariFrameSizeTo200x160(env=env)

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
                use_cuda=args.cuda)

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
        test_process = TestEnvironmentModel(env=env,
                                            environment_model=copy.deepcopy(model),
                                            load_path=save_model_path,
                                            rollout_policy=policy,
                                            args=args)

    trainer = StateSpaceModelTrainer(args=args, env=env, model=model, policy=policy,
                                     optimizer=optimizer,
                                     save_model_path = save_model_path)
    #import time
    #time.sleep(100000000)
    #trainer.train_env_model_batchwise(episoden=100000)

    if args.latent_space_model == "dSSM_DET":
        trainer.train_dSSM(episoden=args.num_episodes, T=10)
    else:
        trainer.train_sSSM(episoden=args.num_episodes, T=10)


class StateSpaceModelTrainer():
    def __init__(self, args, env, model, policy, optimizer, save_model_path):
        self.model = model
        self.args = args
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.use_cuda = args.cuda
        self.batch_size = args.batch_size
        self.sample_memory_size = args.sample_memory_size
        self.log_path = '{0}{1}_{2}.log'.format(args.save_environment_model_dir,
                                                args.env_name,
                                                args.latent_space_model)

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


    def train_dSSM(self, episoden = 1000, T=10, initial_context_size=3, policy_frame_stack=4):
        from collections import deque
        print("create training data")
        create_n_samples = min(self.batch_size * 2, self.sample_memory_size)
        sample_memory = deque(maxlen=self.sample_memory_size)
        sample_memory.extend(self.create_x_samples_T_steps(create_n_samples, T, initial_context_size=initial_context_size, policy_frame_stack=policy_frame_stack))
        criterion = torch.nn.BCELoss()

        for i_episode in range(episoden):
            # sample a state, next-state pair randomly from replay memory for a training step
            sample_observation_initial_context, sample_action_T, sample_next_observation_T, sample_reward_T = [torch.cat(a) for a in zip
                (*random.sample(sample_memory, self.batch_size))]

            image_log_probs, reward = self.model.forward_multiple(sample_observation_initial_context, sample_action_T)

            loss = criterion(image_log_probs, sample_next_observation_T)

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
                sample_memory.extend(self.create_x_samples_T_steps(create_n_samples, T, initial_context_size))




    def train_sSSM(self, episoden = 1000, T=10, initial_context_size = 3, policy_frame_stack=4):
        from collections import deque
        print("create training data")
        create_n_samples = min(self.batch_size * 2, self.sample_memory_size)
        sample_memory = deque(maxlen=self.sample_memory_size)
        sample_memory.extend(self.create_x_samples_T_steps(create_n_samples, T, initial_context_size=initial_context_size, policy_frame_stack=policy_frame_stack))
        criterion = torch.nn.BCELoss()

        for i_episode in range(episoden):
            # sample a state, next-state pair randomly from replay memory for a training step
            sample_observation_initial_context, sample_action_T, sample_next_observation_T, sample_reward_T = [torch.cat(a) for a in zip
                (*random.sample(sample_memory, self.batch_size))]

            #sample_next_observation_T = torch.clamp(sample_next_observation_T, 0.00000001, 1)

            image_log_probs, reward_log_probs, \
            (total_z_mu_prior, total_z_sigma_prior, total_z_mu_posterior, total_z_sigma_posterior) \
                    = self.model.forward_multiple(sample_observation_initial_context, sample_action_T)

            '''for b in range(self.batch_size):
                for i in range(T):
                    #logits = torch.log(image_log_probs[b,i])
                    ##predicted_bernoulli = torch.distributions.bernoulli.Bernoulli(probs=image_log_probs[b,i])
                    #reconstruction = predicted_bernoulli.log_prob(observation[:,0]).mean()
                    #reconstruction = image_log_probs[b,i].mean()

                    reconstruction_loss = criterion(image_log_probs[b,i], sample_next_observation_T[b,i])

                    # begin test
                    prior_gaussian = Normal(loc=total_z_mu_prior[b,i], scale=total_z_sigma_prior[b,i])
                    #prior_log_prob = prior_gaussian.log_prob(prior_gaussian.mean).mean()

                    posterior_gaussian = Normal(loc=total_z_mu_posterior[b,i], scale=total_z_sigma_posterior[b,i])
                    #posterior_log_prob = posterior_gaussian.log_prob(posterior_gaussian.mean).mean()

                    kl_div_loss = torch.distributions.kl.kl_divergence(prior_gaussian, posterior_gaussian)

                    elbo += reconstruction_loss + kl_div_loss.mean()

                    #print(reconstruction_loss, "   ", kl_div_loss.mean())
            '''
            reconstruction_loss = criterion(image_log_probs, sample_next_observation_T)

            prior_gaussian = Normal(loc=total_z_mu_prior, scale=total_z_sigma_prior)
            posterior_gaussian = Normal(loc=total_z_mu_posterior, scale=total_z_sigma_posterior)
            kl_div_loss = torch.distributions.kl.kl_divergence(prior_gaussian, posterior_gaussian)
            loss = reconstruction_loss + kl_div_loss.mean() #loss is elbo

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
                sample_memory.extend(self.create_x_samples_T_steps(create_n_samples, T, initial_context_size=initial_context_size))






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



    def create_x_samples_T_steps(self, number_of_samples, T, initial_context_size, policy_frame_stack):
        from collections import deque
        sample_memory = deque(maxlen=number_of_samples)

        while len(sample_memory) < number_of_samples:
            state = self.env.reset()
            state = torch.from_numpy(state).unsqueeze(0).float()

            frame_stack = state.repeat(1,policy_frame_stack,1,1)

            if self.use_cuda:
                state = state.cuda()
                frame_stack = frame_stack.cuda()

            done = False
            while not done:
                initial_context_stack = None
                action_stack = None
                reward_stack = None
                target_state_stack = None

                for i in range(initial_context_size):
                    value, action, _, _ = self.policy.act(inputs=state, states=None, masks=None)  # no state and mask
                    if initial_context_stack is not None:
                        initial_context_stack = torch.cat((initial_context_stack, state))
                    else:
                        initial_context_stack = state
                    state, reward, done, _ = self.do_env_step(action=action)

                for i in range(T):
                    # let policy decide on next action and perform it
                    value, action, _, _ = self.policy.act(inputs=state, states=None, masks=None)  #no state and mask
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
        if self.use_cuda:
            next_state = next_state.cuda()
            reward = reward.cuda()
        return  next_state, reward, done, info


if __name__ == "__main__":
    main()