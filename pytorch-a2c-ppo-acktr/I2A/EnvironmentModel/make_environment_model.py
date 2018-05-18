import torch
from torch.autograd import Variable
import torch.nn.functional as F
import gym
import gym_minipacman
from I2A.EnvironmentModel.MiniPacmanEnvModel import MiniPacmanEnvModel
from I2A.EnvironmentModel.EnvironmentModelOptimizer import EnvironmentModelOptimizer
from I2A.EnvironmentModel.RenderTrainEM import RenderTrainEM
from logger import LogTrainEM
from custom_envs import make_custom_env
import random

from A2C_Models.I2A_MiniModel import I2A_MiniModel
import argparse
import numpy as np

def main():
    args_parser = argparse.ArgumentParser(description='Make Environment Model arguments')
    args_parser.add_argument('--load_environment_model', action='store_true', default=False,
                             help='flag to continue training on pretrained env_model')
    args_parser.add_argument('--save_environment_model_dir', default="../../trained_models/environment_models/",
                             help='relative path to folder from which a environment model should be loaded.')
    args_parser.add_argument('--load_environment_model_file_name', default="RegularMiniPacman_EnvModel_0.dat",
                             help='file name of the environment model that should be loaded.')
    args_parser.add_argument('--load-policy_dir', default='../../trained_models/a2c',
                             help='directory to save agent logs (default: ./trained_models/)')
    args_parser.add_argument('--load-policy', action='store_true', default=False,
                             help='use trained policy model for environment model training')
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
    args = args_parser.parse_args()

    #args.save_environment_model_dir = os.path.join('../../', 'trained_models/environment_models/')
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    env = make_custom_env(args.env_name, seed=1, rank=1, log_dir=None, grey_scale=True)() #wtf

    policy = build_policy(env=env, use_cuda=args.cuda)

    #relative_load_environment_model_dir = os.path.join('../../', args.load_environment_model_dir)
    environment_model = build_em_model(env=env,
                                       load_environment_model=args.load_environment_model,
                                       load_environment_model_dir=args.save_environment_model_dir,# relative_load_environment_model_dir,
                                       environment_model_file_name=args.load_environment_model_file_name,
                                       use_cuda=args.cuda)

    optimizer = EnvironmentModelOptimizer(model=environment_model, use_cuda=args.cuda)
    optimizer.set_optimizer()

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
    else:
        viz = None

    renderer = RenderTrainEM() if args.render==True else None

    logger = LogTrainEM(log_name="em_trainer_" + args.env_name +".log",
                        delete_log_file = args.load_environment_model==False,
                        viz = viz)
    train_env_model_batchwise(env=env,
                              policy=policy,
                              optimizer=optimizer,
                              use_cuda=args.cuda,
                              renderer=renderer,
                              loss_printer=logger)



def sample_action_from_distribution(actor, action_space, chance_of_random_action=0.25):
    prob = F.softmax(actor, dim=1)
    action = prob.multinomial().data
    use_cuda = action.is_cuda

    if random.random() < chance_of_random_action:
        action_int = random.randint(0, action_space - 1)
        action = torch.from_numpy(np.array([action_int])).unsqueeze(0)
        if use_cuda:
            action = action.cuda()

    action = Variable(action)
    return action

'''
def do_step(env, state, action, use_cuda=False):
    next_state, reward, done, _ = env.step(action.data[0][0])
    next_state = Variable(torch.from_numpy(next_state[-1]).view(state.data.shape)).float()

    reward = Variable(torch.from_numpy(np.array([reward]))).float()
    if use_cuda:
        next_state = next_state.cuda()
        reward = reward.cuda()
    return state, action, next_state, reward
'''


def train_env_model(env, policy, optimizer, use_cuda, renderer = None, loss_printer= None):

    chance_of_random_action = 0.25

    for i_episode in range(10000):
        # loss_printer.reset()
        state = env.reset()
        state = Variable(torch.from_numpy(state).unsqueeze(0)).float()
        if use_cuda:
            state = state.cuda()

        done = False
        while not done:
            # let policy decide on next action and perform it
            critic, actor = policy(state)
            action = sample_action_from_distribution(actor=actor, action_space=env.action_space.n,
                                                     chance_of_random_action=chance_of_random_action)
            next_state, reward, done, _ = env.step(action.data[0][0])
            next_state = Variable(torch.from_numpy(next_state[-1]).view(state.data.shape)).float()

            reward = Variable(torch.from_numpy(np.array([reward]))).float()
            if use_cuda:
                next_state = next_state.cuda()
                reward = reward.cuda()

            loss, prediction = optimizer.optimizer_step(env_state_frame=state,
                                                        env_action=action,
                                                        env_state_frame_target=next_state,
                                                        env_reward_target=reward)
            state = next_state

            # Log, plot and render training
            (predicted_next_state, predicted_reward) = prediction
            if renderer:
                renderer.render_observation(state[0], predicted_next_state[0])

            # log and print infos
            if loss_printer:
                loss_printer.log_loss_and_reward(loss, predicted_reward, reward)
                if loss_printer.frames % 100 == 0:
                    loss_printer.print_episode(episode=i_episode)

def train_env_model_batchwise(env, policy, optimizer, use_cuda, renderer = None, loss_printer= None):
    from collections import deque
    sample_memory = deque(maxlen=2000)

    chance_of_random_action = 0.25
    for i_episode in range(10000):
        #loss_printer.reset()
        state = env.reset()
        state = Variable(torch.from_numpy(state).unsqueeze(0)).float()
        if use_cuda:
            state = state.cuda()

        done = False
        while not done:
            # let policy decide on next action and perform it
            critic, actor = policy(state)
            action = sample_action_from_distribution(actor=actor, action_space=env.action_space.n, chance_of_random_action=chance_of_random_action)
            next_state, reward, done, _ = env.step(action.data[0][0])

            next_state = Variable(torch.from_numpy(next_state[-1]).view(state.data.shape)).float()

            reward = Variable(torch.from_numpy(np.array([reward]))).float()
            if use_cuda:
                next_state = next_state.cuda()
                reward = reward.cuda()

            # add current state, next-state pair to replay memory
            sample_memory.append((state, action, next_state, reward))

            # sample a state, next-state pair randomly from replay memory for a training step
            sample_state, sample_action, sample_next_state, sample_reward = random.choice(sample_memory)
            loss, prediction = optimizer.optimizer_step(env_state_frame = sample_state,
                                                        env_action = sample_action,
                                                        env_state_frame_target = sample_next_state,
                                                        env_reward_target = sample_reward)


            state = next_state

            # Log, plot and render training
            (predicted_next_state, predicted_reward) = prediction
            if renderer:
                renderer.render_observation(sample_state[0], predicted_next_state[0])

            # log and print infos
            if loss_printer:
                loss_printer.log_loss_and_reward(loss, predicted_reward, reward)
                if loss_printer.frames % 100 == 0:
                    loss_printer.print_episode(episode=i_episode)


def save_environment_model(save_model_dir, environment_model_name, environment_model):
    state_to_save = environment_model.state_dict()
    save_model_path = '{0}{1}.dat'.format(save_model_dir, environment_model_name)
    #print(os.path.abspath(save_model_path))
    torch.save(state_to_save, save_model_path)


def build_policy(env, use_cuda):
    # TODO: give option to load policy
    #load_policy_model_dir = os.path.join(root_path, load_policy_model_dir)
    # policy = A2C_PolicyWrapper(load_policy(load_policy_model_dir,
    #                     policy_model,
    #                     action_space=action_space,
    #                     use_cuda=use_cuda,
    #                     policy_name='MiniModel'))
    # Temporary comment: if the next line breaks try it with obs_shape=(1,..,..)
    policy = I2A_MiniModel(obs_shape=env.observation_space.shape, action_space=env.action_space.n, use_cuda=use_cuda)
    if use_cuda:
        policy.cuda()
    return policy

def build_em_model(env, load_environment_model=False, load_environment_model_dir=None, environment_model_file_name=None, use_cuda=True):
    #TODO: @future self: once we have latent space models change the next line
    EMModel = MiniPacmanEnvModel

    reward_bins = env.unwrapped.reward_bins #[0., 1., 2., 5., 0.] for regular


    environment_model = EMModel(obs_shape=env.observation_space.shape,  # 4
                                num_actions=env.action_space.n,
                                reward_bins=reward_bins,
                                use_cuda=use_cuda)

    if load_environment_model:
        saved_state = torch.load('{0}{1}'.format(
            load_environment_model_dir, environment_model_file_name), map_location=lambda storage, loc: storage)
        environment_model.load_state_dict(saved_state)

    if use_cuda:
        environment_model.cuda()

    return environment_model

if __name__ == '__main__':
    main()