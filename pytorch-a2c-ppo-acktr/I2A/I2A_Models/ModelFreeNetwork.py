import torch
import torch.nn as nn
import torch.nn.functional as F
from I2A.utils import get_linear_dims_after_conv

# see B.1: model free path uses identical network as the standard model-free baseline agent (withput the fc layers)
class ModelFreeNetworkMiniPacman(nn.Module):
    def __init__(self, obs_shape, num_outputs = 512):
        super(ModelFreeNetworkMiniPacman, self).__init__()
        input_channels = obs_shape[0]
        input_dims = obs_shape[1:]

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)      #size-preserving
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)                  #19 input -> 10 output (for minipacman)

        linear_input_size = get_linear_dims_after_conv([self.conv1, self.conv2], input_dims)
        self.fc = nn.Linear(linear_input_size, num_outputs)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x