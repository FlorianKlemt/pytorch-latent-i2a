import math

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model import Policy
from I2A.utils import get_linear_dims_after_conv

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

def xavier_weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

class I2A_MiniModel(nn.Module):
    def __init__(self, obs_shape, action_space, use_cuda):     #use_cuda is not used and for compatibility reasons (I2A needs the use_cuda parameter)
        super(I2A_MiniModel, self).__init__()

        input_channels = obs_shape[0]
        input_dims = obs_shape[1:]

        self.conv1 = nn.Conv2d(input_channels, 16, 3, stride=1) #17x17
        self.conv2 = nn.Conv2d(16, 16, 3, stride=2) #8x8

        self.linear_input_size = get_linear_dims_after_conv([self.conv1, self.conv2], input_dims)
        #self.linear_input_size = 16*8*8

        self.linear1 = nn.Linear(self.linear_input_size, 256)

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, action_space)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        #original version
        #self.apply(weights_init)
        #relu_gain = nn.init.calculate_gain('relu')
        #self.conv1.weight.data.mul_(relu_gain)
        #self.conv2.weight.data.mul_(relu_gain)
        #self.linear1.weight.data.mul_(relu_gain)

        self.apply(xavier_weights_init)
        self.conv1.weight.data.mul_(math.sqrt(2))  # Multiplier for relu, only for xavier
        self.conv2.weight.data.mul_(math.sqrt(2))
        self.linear1.weight.data.mul_(math.sqrt(2))
        self.critic_linear.weight.data.mul_(math.sqrt(2))
        self.actor_linear.weight.data.mul_(math.sqrt(2))
        #(x.weight.data.mul_(math.sqrt(2)) for x in [self.conv1, self.conv2, self.linear1, self.critic_linear, self.actor_linear])  # same as above?, but not sure

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))

        x = F.relu(self.conv2(x))

        x = x.view(-1, self.linear_input_size)
        x = F.relu(self.linear1(x))

        return self.critic_linear(x), self.actor_linear(x)
