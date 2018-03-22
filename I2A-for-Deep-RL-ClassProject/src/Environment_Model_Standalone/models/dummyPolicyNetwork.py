import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

class DummyPolicy(nn.Module):

    def __init__(self, num_actions=6):
        super(DummyPolicy, self).__init__()
        self.num_actions = num_actions
        print("Initialized Dummy EnvModel")

    def forward(self, state):
        # Returns a boolean array where the
        summed_states = torch.sum(state[:self.num_actions], dim=0)
        return summed_states == summed_states.max()



