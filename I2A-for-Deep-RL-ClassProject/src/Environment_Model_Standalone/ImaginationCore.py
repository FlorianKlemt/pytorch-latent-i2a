import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from models.dummyPolicyNetwork  import DummyPolicy
from models.environment_model.dummyEnv  import DummyEnv

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

class ImaginationCore(nn.Module):

    def __init__(self, env=None, policy=None):
        super(ImaginationCore, self).__init__()
        self.env = env
        self.policy = policy
        if policy != None:
            self.num_actions = policy.num_actions

    def forward(self, state, action=None):

        if action == None:
            action = self.policy(state)
        elif action > self.num_actions:
            raise IndexError("You passed a to large action id." +
                             "Expected to be < {}, but was {}" \
                             .format(self.num_actions, action))
        elif type(action) == int:
            np_action = np.zeros(self.num_actions)
            np_action[action] = 1
            action = Variable(torch.from_numpy(np_action))
            
        state, reward = self.env(state, action)

        return state, reward


class DummyImagCore(ImaginationCore):

    def __init__(self):
        super(DummyImagCore, self).__init__()
        self.env = DummyEnv()
        self.policy = DummyPolicy()
        self.num_actions = self.policy.num_actions