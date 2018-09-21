import torch.nn as nn
import torch.nn.functional as F
from i2a.utils import get_linear_dims_after_conv
from model_helpers.model_initialization import xavier_weights_init_relu


class RolloutPolicy(nn.Module):
    def __init__(self, obs_shape, action_space):
        super(RolloutPolicy, self).__init__()

        input_channels = obs_shape[0]
        input_dims = obs_shape[1:]

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)

        self.linear_input_size = get_linear_dims_after_conv([self.conv1, self.conv2, self.conv3], input_dims)

        self.linear1 = nn.Linear(self.linear_input_size, 256)

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, action_space)

        self.apply(xavier_weights_init_relu)
        self.train()


    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.linear_input_size)
        x = F.relu(self.linear1(x))

        value = self.critic_linear(x)
        actor = self.actor_linear(x)
        return value, actor

    def sample(self, x):
        probs = F.softmax(x, dim=1)
        action = probs.multinomial(num_samples=1)
        return action

    def action_log_probs(self, actor, action):
        log_probs = F.log_softmax(actor, dim=1)
        action_log_probs = log_probs.gather(1, action)
        return action_log_probs

