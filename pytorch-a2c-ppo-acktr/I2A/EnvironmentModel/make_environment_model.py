import torch
from torch.autograd import Variable
import torch.nn.functional as F
import gym
import gym_minipacman
from I2A.EnvironmentModel.MiniPacmanEnvModel import MiniPacmanEnvModel
from I2A.EnvironmentModel.EnvironmentModelOptimizer import EnvironmentModelOptimizer
from I2A.EnvironmentModel.RenderTrainEM import RenderTrainEM
from custom_envs import make_custom_env
import os

from A2C_Models.I2A_MiniModel import I2A_MiniModel
import argparse


def main():
    args_parser = argparse.ArgumentParser(description='Make Environment Model arguments')
    args_parser.add_argument('--load_environment_model', action='store_true', default=False,
                             help='flag to continue training on pretrained env_model')
    args_parser.add_argument('--load_environment_model_dir', default="trained_models/environment_models_trained/",
                             help='relative path to folder from which a environment model should be loaded.')
    args_parser.add_argument('--load_environment_model_file_name', default="RegularMiniPacman_EnvModel_0.dat",
                             help='file name of the environment model that should be loaded.')
    args = args_parser.parse_args()

    save_environment_model_dir = os.path.join('../../', 'trained_models/environment_models/')

    train_minipacman(args=args,
             env_name="RegularMiniPacmanNoFrameskip-v0",
             policy_model="RegularMiniPacmanNoFrameskip-v0.pt",
             load_policy_model_dir="trained_models/a2c/",
             environment_model_name="RegularMiniPacman_EnvModel_trained",
             save_environment_model_dir=save_environment_model_dir,
             use_cuda=True)

def train_minipacman(args, env_name="RegularMiniPacmanNoFrameskip-v0",
             policy_model = None,   #TODO: needs to be passed to build_em_model()
             load_policy_model_dir = None,  #TODO: same
             environment_model_name = "pong_em",
             save_environment_model_dir = "trained_models/environment_models_trained/",
             render=True,
             use_cuda=False):

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    env = make_custom_env(env_name, seed=1, rank=1, log_dir=None)() #wtf

    policy = build_policy(env=env, use_cuda=use_cuda)

    relative_load_environment_model_dir = os.path.join('../../', args.load_environment_model_dir)
    environment_model = build_em_model(env=env,
                                       load_environment_model=args.load_environment_model,
                                       load_environment_model_dir=relative_load_environment_model_dir,
                                       environment_model_file_name=args.load_environment_model_file_name,
                                       use_cuda=use_cuda)

    optimizer = EnvironmentModelOptimizer(model=environment_model, use_cuda=use_cuda)
    optimizer.set_optimizer()

    chance_of_random_action = 0.25

    if render==True:
        renderer = RenderTrainEM(environment_model_name, delete_log_file = args.load_environment_model==False)

    for i_episode in range(10000):
        print("Start episode ",i_episode)

        state = env.reset()

        done = False
        sum_reward = 0

        while not done:
            state_variable = Variable(torch.from_numpy(state).unsqueeze(0).type(FloatTensor))
            critic, actor = policy(state_variable)

            prob = F.softmax(actor, dim=1)
            action = prob.multinomial()
            action_int = action.data[0][0]
            #if random.random() < chance_of_random_action:
            #    action = random.randint(0, env.action_space.n -1)


            next_state, reward, done, _ = env.step(action_int)

            next_state_variable = torch.from_numpy(next_state[-1]).type(FloatTensor)
            next_state_variable = Variable(next_state_variable.unsqueeze(0))

            reward = Variable(FloatTensor([reward]))

            loss, prediction = optimizer.optimizer_step(state_variable,
                                                        action,
                                                        next_state_variable,
                                                        reward)

            (predicted_next_state, predicted_reward) = prediction
            state = next_state


            if render:
                renderer.render_observation(next_state_variable, predicted_next_state[0])

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

    #TODO: change this depending on the env
    reward_bins = [0., 1., 2., 5., 0.]


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