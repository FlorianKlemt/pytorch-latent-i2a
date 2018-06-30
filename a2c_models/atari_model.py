import torch.nn as nn
import torch.nn.functional as F
from utils import init

class AtariModel(nn.Module):
    def __init__(self,  obs_shape, action_space, use_cuda):     #use_cuda is not used and for compatibility reasons (I2A needs the use_cuda parameter)
        super(AtariModel, self).__init__()
        from i2a.utils import get_linear_dims_after_conv

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        input_channels = obs_shape[0]
        input_dims = obs_shape[1:]

        self.conv1 = init_(nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0))
        self.conv2 = init_(nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0))
        self.conv3 = init_(nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0))
        self.conv4 = init_(nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0))

        self.linear_input_size = get_linear_dims_after_conv(
            [self.conv1, self.conv2, self.conv3, self.conv4], input_dims)
        # self.linear_input_size = 16*8*8

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.linear1 = init_(nn.Linear(self.linear_input_size, 256))

        self.critic_linear = init_(nn.Linear(256, 1))
        self.actor_linear = init_(nn.Linear(256, action_space))

        self.train()

    def forward(self, inputs):
        x = F.leaky_relu(self.conv1(inputs))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))

        x = x.view(-1, self.linear_input_size)
        x = F.relu(self.linear1(x))

        return self.critic_linear(x), self.actor_linear(x)

