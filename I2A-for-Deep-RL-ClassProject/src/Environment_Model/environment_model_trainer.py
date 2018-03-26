import os

import torch
from torch.autograd import Variable
import numpy as np
import cv2
import torch.nn.functional as F
import logging
import time
import random
from Environment_Model.environment_model import EnvironmentModelOptimizer
from Environment_Model.load_environment_model import load_em_model
from Environment_Model.load_preprocessed_atari_environment import load_atari_environment
from Environment_Model.load_small_a3c_policy_model import load_policy


def get_step_summary(episode, iteration,
                     frame_loss, reward_loss,
                     pred_reward, reward,
                     duration, total_time):


    iteration_suffix = ''
    if iteration >= 1000:
        iteration /= 1000
        iteration_suffix = 'k'

    str_game = 'Episode: {0:.0f}\t'.format(episode)
    str_iteration = 'it: {0:.1f}{1}\t\t'.format(iteration, iteration_suffix)
    str_loss = 'frame loss: {0:.9f}\t reward loss: {1:.5f}\t\t'.format(frame_loss, reward_loss)
    str_reward = 'pred reward: {0:.3f}\t reward: {1:.3f}\t\t'.format(pred_reward, reward)
    str_time = 'duration: {0:.3f}s\t total duration{1:.3f}min'.format(duration, total_time/60)
    return str_game + str_iteration + str_loss + str_reward + str_time



class RenderTrainEM():
    def __init__(self, log_name, delete_log_file = True):
        render_window_sizes = (400, 400)
        cv2.namedWindow('target', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('target', render_window_sizes)
        cv2.namedWindow('prediction', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('prediction', render_window_sizes)
        #cv2.namedWindow('differnece', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('differnece', render_window_sizes)

        # this logger will both print to the console as well as the file

        log_file = os.path.join('logs', 'em_trainer_' + log_name +'.log')
        #log_file = 'logs/em_trainer_log.log'
        if delete_log_file == True and os.path.exists(log_file):
            os.remove(log_file)
        self.logger_prediction = logging.getLogger('em_trainer_log')
        self.logger_prediction.addHandler(logging.FileHandler(log_file))
        self.logger_prediction.setLevel(logging.INFO)
        self.iteration = 0
        self.start_time = time.time()
        self.train_total_time = 0

    def render_observation_in_window(self, window_name, observation):
        drawable_state = observation.data[0]
        drawable_state = np.swapaxes(drawable_state, 0, 1)
        drawable_state = np.swapaxes(drawable_state, 1, 2)

        drawable_state = drawable_state.numpy()

        frame_data = (drawable_state * 255.0)

        frame_data[frame_data < 0] = 0
        frame_data[frame_data > 255] = 255
        frame_data = frame_data.astype(np.uint8)

        cv2.imshow(window_name, frame_data)
        cv2.waitKey(1)


    def render_observation(self, target, prediction):
        self.render_observation_in_window('target', target)
        self.render_observation_in_window('prediction', prediction)
        #loss = (target - prediction) + 0.5
        #self.render_observation_in_window('differnece', loss)

    def log_loss_and_reward(self, episode, state_loss, reward_loss,
                            reward_prediction, reward):
        self.iteration += 1
        duration = time.time() - self.start_time
        self.start_time = time.time()
        self.train_total_time += duration

        state_loss = state_loss.data.cpu().numpy()[0]
        reward_loss = reward_loss.data.cpu().numpy()[0]
        reward_prediction = reward_prediction.data.cpu().numpy()[0][0]
        reward = reward.data.cpu().numpy()[0]

        summary = get_step_summary(episode, self.iteration,
                                   state_loss, reward_loss,
                                   reward_prediction, reward,
                                   duration, self.train_total_time)
        self.logger_prediction.info(summary)
        if self.iteration % 10 == 0 or reward < -0.9 or reward > 0.9:
            print(summary)



def save_environment_model(save_model_dir, environment_model_name, environment_model):
    state_to_save = environment_model.state_dict()
    save_model_path = '{0}{1}.dat'.format(save_model_dir, environment_model_name)
    torch.save(state_to_save, save_model_path)

## trainer ####################################################################

def train_em(atari_env="PongDeterministic-v4",
             EMModel = None,
             policy_model = "PongDeterministic-v4_21",
             load_policy_model_dir = "trained_models/",
             environment_model_name = "pong_em",
             save_environment_model_dir = "trained_models/environment_models/",
             load_environment_model = False,
             load_environment_model_dir="trained_models/environment_models/",
             root_path="",
             use_cuda=False):

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    env = load_atari_environment(atari_env)
    action_space = env.action_space.n

    load_policy_model_dir = os.path.join(root_path, load_policy_model_dir)
    policy = load_policy(load_policy_model_dir,
                         policy_model,
                         action_space=action_space,
                         use_cuda=use_cuda)

    save_environment_model_dir = os.path.join(root_path, save_environment_model_dir)
    if load_environment_model:
        load_environment_model_dir = os.path.join(root_path, load_environment_model_dir)
        environment_model = load_em_model(EMModel,
                                          load_environment_model_dir,
                                          environment_model_name,
                                          action_space,
                                          use_cuda)
    else:
        environment_model = EMModel(name=environment_model_name,
                                    num_input_actions=env.action_space.n,
                                    use_cuda=use_cuda)

    if use_cuda:
        environment_model.cuda()

    optimizer = EnvironmentModelOptimizer(model=environment_model,
                                          lstm_backward_steps= 3,
                                          use_cuda=use_cuda)
    optimizer.set_optimizer()

    chance_of_random_action = 0.25

    render = True

    renderer = RenderTrainEM(environment_model_name, delete_log_file = load_environment_model==False)

    for i_episode in range(10000):
        print("Start episode ",i_episode)
        policy.repackage_lstm_hidden_variables()

        state = env.reset()
        state = torch.from_numpy(state).type(FloatTensor)
        state = Variable(state.unsqueeze(0), requires_grad=False)

        done = False
        sum_reward = 0

        while not done:

            critic, actor = policy(state)

            prob = F.softmax(actor, dim=1)
            action = prob.multinomial().data
            if random.random() < chance_of_random_action:
                action = random.randint(0, env.action_space.n -1)

            next_state, reward, done, _ = env.step(action)
            next_state = torch.from_numpy(next_state).type(FloatTensor)
            next_state = Variable(next_state.unsqueeze(0))

            reward = Variable(FloatTensor([reward]))


            np_action = np.zeros(env.action_space.n)
            np_action[action] = 1
            action = Variable(torch.from_numpy(np_action)).type(FloatTensor)

            loss, prediction = optimizer.optimizer_step(state,
                                                        action,
                                                        next_state,
                                                        reward)

            (predicted_next_state, predicted_reward) = prediction
            state = next_state

            if render:
                renderer.render_observation(next_state, predicted_next_state)

            # log and print infos
            (next_state_loss, next_reward_loss) = loss
            renderer.log_loss_and_reward(i_episode, next_state_loss,
                                         next_reward_loss,
                                         predicted_reward,
                                         reward)

            r = reward.data.cpu().numpy()[0]
            if r > 0.9 or r < -0.9:
                sum_reward += r
                #environment_model.repackage_lstm_hidden_variables()
                print("Reward", r, "total reward", sum_reward)

        #environment_model.repackage_lstm_hidden_variables()
        print("Save model", save_environment_model_dir, environment_model_name)
        save_environment_model(save_model_dir = save_environment_model_dir,
                               environment_model_name = environment_model_name,
                               environment_model = environment_model)







#experimental
def train_minipacman(env_name="RegularMiniPacman-v0",
             EMModel = None,
             policy_model = "PongDeterministic-v4_21",
             load_policy_model_dir = "trained_models/",
             environment_model_name = "pong_em",
             save_environment_model_dir = "trained_models/environment_models/",
             load_environment_model = False,
             load_environment_model_dir="trained_models/environment_models/",
             root_path="",
             use_cuda=False):

    import gym
    import gym_minipacman
    from A3C_model import SmallA3Clstm

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    env = gym.make(env_name)
    action_space = env.action_space.n

    #load_policy_model_dir = os.path.join(root_path, load_policy_model_dir)
    #policy = load_policy(load_policy_model_dir,
    #                     policy_model,
    #                     action_space=action_space,
    #                     use_cuda=use_cuda)
    policy = SmallA3Clstm(num_inputs=3, action_space=action_space, use_cuda=use_cuda)
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
        environment_model = EMModel(num_inputs = 3,
                                    num_actions=env.action_space.n,
                                    use_cuda=use_cuda)

    if use_cuda:
        environment_model.cuda()

    optimizer = EnvironmentModelOptimizer(model=environment_model,
                                          lstm_backward_steps= 3,
                                          use_cuda=use_cuda)
    optimizer.set_optimizer()

    chance_of_random_action = 0.25

    render = True

    renderer = RenderTrainEM(environment_model_name, delete_log_file = load_environment_model==False)

    for i_episode in range(10000):
        print("Start episode ",i_episode)
        #policy.repackage_lstm_hidden_variables()   #only for policies with lstm

        state = env.reset()
        state = torch.from_numpy(state).type(FloatTensor)
        state = Variable(state.unsqueeze(0), requires_grad=False)

        done = False
        sum_reward = 0

        while not done:

            critic, actor = policy(state)

            prob = F.softmax(actor, dim=1)
            action = prob.multinomial().data
            if random.random() < chance_of_random_action:
                action = random.randint(0, env.action_space.n -1)

            next_state, reward, done, _ = env.step(action)
            next_state = torch.from_numpy(next_state).type(FloatTensor)
            next_state = Variable(next_state.unsqueeze(0))

            reward = Variable(FloatTensor([reward]))


            np_action = np.zeros(env.action_space.n)
            np_action[action] = 1
            action = Variable(torch.from_numpy(np_action)).type(FloatTensor)

            loss, prediction = optimizer.optimizer_step(state,
                                                        action,
                                                        next_state,
                                                        reward)

            (predicted_next_state, predicted_reward) = prediction
            state = next_state

            if render:
                renderer.render_observation(next_state, predicted_next_state)

            # log and print infos
            (next_state_loss, next_reward_loss) = loss
            renderer.log_loss_and_reward(i_episode, next_state_loss,
                                         next_reward_loss,
                                         predicted_reward,
                                         reward)

            r = reward.data.cpu().numpy()[0]
            if r > 0.9 or r < -0.9:
                sum_reward += r
                #environment_model.repackage_lstm_hidden_variables()
                print("Reward", r, "total reward", sum_reward)

        #environment_model.repackage_lstm_hidden_variables()
        print("Save model", save_environment_model_dir, environment_model_name)
        save_environment_model(save_model_dir = save_environment_model_dir,
                               environment_model_name = environment_model_name,
                               environment_model = environment_model)

