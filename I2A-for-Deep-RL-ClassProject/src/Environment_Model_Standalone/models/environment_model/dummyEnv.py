import torch
from torch import nn

class DummyEnv(nn.Module):

    def __init__(self):
        super(DummyEnv, self).__init__()

    def forward(self, state, action):
        inc_state = torch.add(state, 0.2)
        inc_reward = torch.add(action, 2)
        return inc_state, inc_reward


