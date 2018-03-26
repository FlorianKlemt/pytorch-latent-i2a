from model import ActorCritic
from envs import WrapPyTorch
from baselines.common.atari_wrappers import *
import gym
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class RenderAtari():
    def __init__(self, env_id, actor_critic):
        self.actor_critic = actor_critic
        self.env = gym.make(env_id)
        self.env = wrap_deepmind(self.env)
        self.env = WrapPyTorch(self.env)
        self.reset()

    def render(self):
        self.env.render()

    def set_policy(self, actor_critic):
        self.actor_critic = actor_critic

    def step(self):
        value, logits = self.actor_critic(Variable(self.state, volatile=True).cuda())
        probs = F.softmax(logits)
        log_probs = F.log_softmax(logits).data
        action = probs.multinomial().data
        cpu_actions = action.cpu()
        cpu_actions = cpu_actions.numpy()

        self.state, reward, done, info = self.env.step(cpu_actions)
        self.state = self.state_to_torch(self.state)
        self.rewards.append(reward)

        return done

    def state_to_torch(self, state):
        state = np.stack(state)
        state = np.asarray([state, state, state, state])
        state = torch.from_numpy(state).float()
        state = state.permute(1, 0, 2, 3)
        state.cuda()
        return state

    def reset(self):
        self.state = self.env.reset()
        self.state = self.state_to_torch(self.state)
        self.rewards = []