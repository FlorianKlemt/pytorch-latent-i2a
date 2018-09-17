import torch.nn as nn
import torch.nn.functional as F
from i2a.utils import get_linear_dims_after_conv


# see B.1: model free path uses identical network as the standard model-free baseline agent (without the fc layers)
class LatentSpaceModelFreeNetwork(nn.Module):
    def __init__(self, obs_shape, num_outputs=512):
        super(LatentSpaceModelFreeNetwork, self).__init__()
        input_channels = obs_shape[0]
        input_dims = obs_shape[1:]

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0)

        linear_input_size = get_linear_dims_after_conv([self.conv1, self.conv2, self.conv3, self.conv4], input_dims)
        self.fc = nn.Linear(linear_input_size, num_outputs)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x