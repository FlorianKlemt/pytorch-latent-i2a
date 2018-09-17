import torch.nn as nn
import torch.nn.functional as F

class OutputPolicyNetwork(nn.Module):
    def __init__(self, input_size, nr_actions):
        super(OutputPolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, nr_actions)

    def forward(self,x):
        x = F.relu(self.fc(x))
        policy = self.actor_linear(x)
        value = self.critic_linear(x)
        return policy,value
