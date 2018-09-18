import torch.nn as nn
import torch.nn.functional as F
from i2a.utils import get_linear_dims_after_conv

# see B.1: model free path uses identical network as the standard model-free baseline agent (without the fc layers)
class ModelFreeNetwork(nn.Module):
    def __init__(self, obs_shape, num_outputs = 512):
        super(ModelFreeNetwork, self).__init__()

        self._output_size = num_outputs;
        input_channels = obs_shape[0]
        input_dims = obs_shape[1:]

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)

        linear_input_size = get_linear_dims_after_conv([self.conv1, self.conv2], input_dims)
        self.fc = nn.Linear(linear_input_size, num_outputs)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def output_size(self):
        return self._output_size