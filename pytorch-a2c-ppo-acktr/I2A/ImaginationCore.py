import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from I2A.EnvironmentModel.MiniPacmanEnvModel import MiniPacmanEnvModel
from I2A.load_utils import load_policy, load_em_model
from A2C_Models.A2C_PolicyWrapper import A2C_PolicyWrapper
from A2C_Models.I2A_MiniModel import I2A_MiniModel
import os

class ImaginationCore(nn.Module):
    def __init__(self, env_model=None, rollout_policy=None, use_cuda=False):
        super(ImaginationCore, self).__init__()
        self.env_model = env_model
        self.rollout_policy = rollout_policy
        #if rollout_policy != None:
        #    self.num_actions = rollout_policy.policy.action_space
        #self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    def forward(self, state, action):
        next_state, reward = self.env_model(state, action)

        x = torch.cat((state, next_state), 1)
        next_state = x[:, 1:]
        return next_state, reward

    def sample(self, input_state):
        #TODO: HOW THE FUCK IS THIS SUPPOSED TO BE WITH THE RETARDED ACT FUNCTION IN THIS REPO??
        #critic, actor = self.policy(state)
        value, action, action_log_probs, states = self.rollout_policy.act(input_state, None, None)
        #action = action.cpu().data[0][0]
        return action
