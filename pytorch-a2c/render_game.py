from envs import WrapPyTorch
from baselines.common.atari_wrappers import *
import gym
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

import collections

class RenderAtari():
    def __init__(self, env_id, actor_critic):
        self.actor_critic = actor_critic
        self.env = gym.make(env_id)
        self.env = wrap_deepmind(self.env)
        self.env = WrapPyTorch(self.env)
        self.states = collections.deque(maxlen=4)
        self.reset()

    def render(self):
        self.env.render()

    def set_policy(self, actor_critic):
        self.actor_critic = actor_critic

    def step(self):
        states_torch = self.states_to_torch(self.states)
        value, logits = self.actor_critic(Variable(states_torch, volatile=True).cuda())
        probs = F.softmax(logits)
        log_probs = F.log_softmax(logits).data
        action = probs.multinomial().data
        cpu_actions = action.cpu()
        cpu_actions = cpu_actions.numpy()
        cpu_actions = cpu_actions[0][0]

        state, reward, done, info = self.env.step(cpu_actions)
        self.rewards.append(reward)

        return done

    def states_to_torch(self, states):
        states = np.stack(states)
        states = torch.from_numpy(states).float()
        states = states.permute(1, 0, 2, 3)
        states.cuda()
        return states

    def reset(self):
        state = self.env.reset()
        self.states.append(state)
        self.states.append(state)
        self.states.append(state)
        self.states.append(state)
        self.rewards = []