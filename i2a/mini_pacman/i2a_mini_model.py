import torch.nn as nn
import torch.nn.functional as F
from i2a.utils import get_linear_dims_after_conv
from model_helpers.model_initialization import xavier_weights_init


class I2A_MiniModel(nn.Module):
    def __init__(self, obs_shape, action_space, use_cuda):     #use_cuda is not used and for compatibility reasons (I2A needs the use_cuda parameter)
        super(I2A_MiniModel, self).__init__()

        input_channels = obs_shape[0]
        input_dims = obs_shape[1:]

        self.conv1 = nn.Conv2d(input_channels, 16, 3, stride=1) #17x17
        self.conv2 = nn.Conv2d(16, 16, 3, stride=2) #8x8

        self.linear_input_size = get_linear_dims_after_conv([self.conv1, self.conv2], input_dims)

        self.linear1 = nn.Linear(self.linear_input_size, 256)

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, action_space)

        self.apply(xavier_weights_init)
        self.train()

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))

        x = F.relu(self.conv2(x))

        x = x.view(-1, self.linear_input_size)
        x = F.relu(self.linear1(x))

        return self.critic_linear(x), self.actor_linear(x)
