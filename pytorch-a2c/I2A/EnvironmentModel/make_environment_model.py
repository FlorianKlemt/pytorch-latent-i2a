import os
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import gym
import gym_minipacman
from I2A.EnvironmentModel.MiniPacmanEnvModel import MiniPacmanEnvModel
from I2A.EnvironmentModel.EnvironmentModelOptimizer import EnvironmentModelOptimizer
from I2A.EnvironmentModel.RenderTrainEM import RenderTrainEM
from minipacman_envs import make_minipacman_env_no_log
import os
import collections
import sys

from I2A.load_utils import load_policy, load_em_model

#root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
root_dir = "/home/flo/Dokumente/I2A_GuidedResearch/pytorch-a2c/"
#root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

def main():
    EMModel = MiniPacmanEnvModel

    train_minipacman(env_name="RegularMiniPacmanNoFrameskip-v0",
             EMModel=EMModel,
             policy_model="RegularMiniPacmanNoFrameskip-v0.pt",
             load_policy_model_dir="trained_models/a2c/",
             environment_model_name="RegularMiniPacman_EnvModel_0",
             save_environment_model_dir="trained_models/environment_models/",
             load_environment_model=False,
             load_environment_model_dir="trained_models/environment_models/",
             root_path=root_dir,
             use_cuda=True)

def states_to_torch(states, use_cuda):
    states = np.stack(states)
    states = torch.from_numpy(states).float()
    states = states.permute(1, 0, 2, 3)
    if use_cuda:
        states = states.type(torch.cuda.FloatTensor)
        states.cuda()
    states = Variable(states)#, requires_grad=False)
    return states

def train_minipacman(env_name="RegularMiniPacmanNoFrameskip-v0",
             EMModel = None,
             policy_model = "PongDeterministic-v4_21",
             load_policy_model_dir = "trained_models/",
             environment_model_name = "pong_em",
             save_environment_model_dir = "trained_models/environment_models_trained/",
             load_environment_model = False,
             load_environment_model_dir="trained_models/environment_models_trained/",
             root_path="",
             render=True,
             use_cuda=False):

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    env = make_minipacman_env_no_log(env_name)#gym.make(env_name)
    action_space = env.action_space.n

    load_policy_model_dir = os.path.join(root_path, load_policy_model_dir)
    policy = load_policy(load_policy_model_dir,
                         policy_model,
                         action_space=action_space,
                         use_cuda=use_cuda,
                         policy_name='MiniModel')
    #policy = MiniModel(1,action_space)  #1?
    if use_cuda:
        policy.cuda()

    save_environment_model_dir = os.path.join(root_path, save_environment_model_dir)
    if load_environment_model:
        load_environment_model_dir = os.path.join(root_path, load_environment_model_dir)
        environment_model = load_em_model(EMModel,
                                          load_environment_model_dir,
                                          environment_model_name,
                                          action_space,
                                          use_cuda)
    else:
        environment_model = EMModel(num_inputs = 4,
                                    num_actions=env.action_space.n,
                                    use_cuda=use_cuda)

    if use_cuda:
        environment_model.cuda()

    optimizer = EnvironmentModelOptimizer(model=environment_model,
                                          use_cuda=use_cuda)
    optimizer.set_optimizer()

    chance_of_random_action = 0.25

    #num_frames = 4
    #states_deque = collections.deque(maxlen=num_frames)

    if render==True:
        renderer = RenderTrainEM(environment_model_name, delete_log_file = load_environment_model==False)

    for i_episode in range(10000):
        print("Start episode ",i_episode)

        state = env.reset()
        #state = torch.from_numpy(state).type(FloatTensor)
        #state = Variable(state)
        #state = Variable(state.unsqueeze(0), requires_grad=False)

        done = False
        sum_reward = 0

        while not done:
            #states = states_to_torch(states_deque, use_cuda)
            state_variable = Variable(torch.from_numpy(state).type(FloatTensor))
            critic, actor = policy(state_variable)   #was states

            prob = F.softmax(actor, dim=1)
            action = prob.multinomial().data
            action = action[0][0]
            if random.random() < chance_of_random_action:
                action = random.randint(0, env.action_space.n -1)


            next_state, reward, done, _ = env.step(action)

            next_state_variable = torch.from_numpy(next_state[:,-1]).type(FloatTensor)
            next_state_variable = Variable(next_state_variable.unsqueeze(0))

            reward = Variable(FloatTensor([reward]))

            loss, prediction = optimizer.optimizer_step(state_variable,
                                                        action,
                                                        next_state_variable,
                                                        reward)

            (predicted_next_state, predicted_reward) = prediction
            state = next_state


            if render:
                renderer.render_observation(next_state_variable, predicted_next_state)

            # log and print infos
            (next_state_loss, next_reward_loss) = loss
            renderer.log_loss_and_reward(i_episode, next_state_loss,
                                         next_reward_loss,
                                         predicted_reward,
                                         reward)

            r = reward.data.cpu().numpy()[0]
            if r > 0.9 or r < -0.9:
                sum_reward += r
                #print("Reward", r, "total reward", sum_reward)

        print("Save model", save_environment_model_dir, environment_model_name)
        save_environment_model(save_model_dir = save_environment_model_dir,
                               environment_model_name = environment_model_name,
                               environment_model = environment_model)

def save_environment_model(save_model_dir, environment_model_name, environment_model):
    state_to_save = environment_model.state_dict()
    save_model_path = '{0}{1}.dat'.format(save_model_dir, environment_model_name)
    torch.save(state_to_save, save_model_path)


if __name__ == '__main__':
    main()