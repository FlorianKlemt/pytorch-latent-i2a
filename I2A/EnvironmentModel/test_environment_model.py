import os
import torch
from torch.autograd import Variable
import numpy as np
import time
import cv2
import sys
from random import randint
import collections
from collections import deque
from custom_envs import make_custom_env
from I2A.ImaginationCore import ImaginationCore
from A2C_Models.I2A_MiniModel import I2A_MiniModel
from A2C_Models.A2C_PolicyWrapper import A2C_PolicyWrapper
import gym_minipacman

from multiprocessing import Process

class RenderImaginationCore():
    def __init__(self, grey_scale):
        self.grey_scale = grey_scale

        render_window_sizes = (1520, 760)
        self.window_name = 'imagination_core'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, render_window_sizes)


    def render_preprocessing(self, observation, reward_text, step_text):
        #drawable_state = observation.permute(1, 2, 0)
        drawable_state = observation.view(observation.data.shape[1], observation.data.shape[2], -1)

        drawable_state = drawable_state.data.cpu().numpy()

        zeros = np.ones((4, drawable_state.shape[1], drawable_state.shape[2]))
        drawable_state = np.append(drawable_state, zeros, axis=0)
        drawable_state = drawable_state.repeat(40,0).repeat(40,1)

        frame_data = (drawable_state * 255.0)

        frame_data[frame_data < 0] = 0
        frame_data[frame_data > 255] = 255
        frame_data = frame_data.astype(np.uint8)

        cv2.putText(frame_data, reward_text, (20, 720), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)
        cv2.putText(frame_data, step_text, (20, 680), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)

        if not self.grey_scale:
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)

        return frame_data

    def render_observation(self, observation, predicted_observation, reward, predicted_reward, rollout_step):
        frame_data1 = self.render_preprocessing(observation, 'true reward: {0:.3f} '.format(reward), 'rollout step: '+str(rollout_step))
        frame_data2 = self.render_preprocessing(predicted_observation, 'predicted reward: {0:.3f} '.format(predicted_reward), 'rollout step: '+str(rollout_step))

        both = np.hstack((frame_data1, frame_data2))

        cv2.imshow(self.window_name, both)
        cv2.waitKey(1000)



def numpy_to_variable(numpy_value, use_cuda):
    value = Variable(torch.from_numpy(numpy_value).unsqueeze(0), requires_grad=False).float()
    if use_cuda:
        value = value.cuda()
    return value

def play_with_imagination_core(imagination_core, env, args):
    render = True
    renderer = RenderImaginationCore(args.grey_scale)

    #for i_episode in range(20):
    observation = env.reset()
    state = numpy_to_variable(observation, args.cuda)

    # do 20 random actions to get different start observations
    for i in range(randint(20, 50)):
        observation, reward, done, _ = env.step(env.action_space.sample())
        state = numpy_to_variable(observation, args.cuda)

    # todo remove only used for testing rgb to class converter
    from I2A.EnvironmentModel.minipacman_rgb_class_converter import MiniPacmanRGBToClassConverter
    x = MiniPacmanRGBToClassConverter()
    p = x.minipacman_rgb_to_class(state)
    p = x.minipacman_class_to_rgb(p)
    # end remove

    renderer.render_observation(state[0], p[0], reward, reward, 0)
    # render start state
    #if render:
    #    renderer.render_observation('start_state', state[0])

    predicted_state = state

    for t in range(5):
        action = imagination_core.sample(predicted_state)
        predicted_state, predicted_reward = imagination_core(predicted_state, action)

        predicted_reward = predicted_reward.data.cpu().numpy()
        print(t+1, "reward", np.max(predicted_reward[0], 0))

        observation, reward, done, _ = env.step(action.item())
        state = numpy_to_variable(observation, args.cuda)

        if render:
            renderer.render_observation(state[0], predicted_state[0], reward, predicted_reward[0], t+1)


def test_environment_model(env, environment_model, load_path, rollout_policy, args):
    i = 1
    while(True):
        #env_name = "RegularMiniPacmanNoFrameskip-v0"
        #env = make_custom_env(env_name, seed=1, rank=1, log_dir=None, grey_scale=True)()

        # small model which only predicts one reward
        saved_state = torch.load(load_path, map_location=lambda storage, loc: storage)
        environment_model.load_state_dict(saved_state)

        if args.cuda:
            environment_model.cuda()

        imagination_core = ImaginationCore(env_model=environment_model, rollout_policy=rollout_policy,
                                           grey_scale = False, frame_stack = 1)

        print("started game", i)
        play_with_imagination_core(imagination_core, env=env, args=args)

        print("finished game", i)
        time.sleep(5)
        i += 1

class TestEnvironmentModel():
    def __init__(self, env, environment_model, load_path, rollout_policy, args):
        #load_path = os.path.join(load_path, args.env_name + ".pt")
        self.p = Process(target = test_environment_model,
                         args=(env, environment_model, load_path, rollout_policy, args))
        self.p.start()

    def stop(self):
        self.p.terminate()

'''
### Begin Main ###
use_cuda = True
#root_dir = '/home/flo/Dokumente/I2A_GuidedResearch/pytorch-a2c-ppo-acktr/trained_models/environment_models/'    #os.path.dirname(os.path.realpath(sys.argv[0]))


env_name = "RegularMiniPacmanNoFrameskip-v0"
env = make_custom_env(env_name, seed=1, rank=1, log_dir=None, grey_scale=True)()

# small model which only predicts one reward
EMModel = MiniPacmanEnvModel

environment_model = EMModel(obs_shape=env.observation_space.shape,
                                num_actions=env.action_space.n,
                                reward_bins=env.unwrapped.reward_bins,
                                use_cuda=use_cuda)
saved_state = torch.load("../../trained_models/environment_models/RegularMiniPacman_EnvModel_0.dat", map_location=lambda storage, loc: storage)
environment_model.load_state_dict(saved_state)

rollout_policy = A2C_PolicyWrapper(I2A_MiniModel(obs_shape=env.observation_space.shape, action_space=env.action_space.n, use_cuda=use_cuda))
if use_cuda:
    environment_model.cuda()
    rollout_policy.cuda()

imagination_core = ImaginationCore(env_model=environment_model, rollout_policy=rollout_policy)
#imagination_core = ImaginationCore(num_inputs=4, action_space=env.action_space.n,
#                                             em_model_reward_bins=em_model_reward_bins, use_cuda=use_cuda, require_grad=False)    #no policy grads required for this

play_with_imagination_core(imagination_core, env=env, use_cuda=use_cuda)
'''