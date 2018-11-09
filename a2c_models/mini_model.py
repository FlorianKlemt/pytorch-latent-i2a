import math

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model import Policy
from distributions import Categorical

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


def xavier_weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

class MiniModel(Policy):
    def __init__(self, num_inputs, action_space, use_cuda):     #use_cuda is not used and for compatibility reasons (I2A needs the use_cuda parameter)
        super(MiniModel, self).__init__()

        self.dist = Categorical(256, action_space)

        self.conv1 = nn.Conv2d(num_inputs, 16, 3, stride=1) #17x17
        self.conv2 = nn.Conv2d(16, 16, 3, stride=2) #8x8

        self.linear_input_size = 16*8*8

        self.linear1 = nn.Linear(self.linear_input_size, 256)

        self.critic_linear = nn.Linear(256, 1)

        self.apply(weights_init)

        self.conv1.weight.data.mul_(math.sqrt(2))  # Multiplier for relu
        self.conv2.weight.data.mul_(math.sqrt(2))  # Multiplier for relu
        self.linear1.weight.data.mul_(math.sqrt(2))  # Multiplier for relu

        self.train()

    @property
    def state_size(self):
        return 1

    def forward(self, inputs, states=None, masks=None):
        x = F.relu(self.conv1(inputs))

        x = F.relu(self.conv2(x))

        x = x.view(-1, self.linear_input_size)
        x = F.relu(self.linear1(x))
        return x, x, states
