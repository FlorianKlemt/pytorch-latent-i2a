import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init

class OutputPolicyNetwork(nn.Module):
    def __init__(self, input_size, nr_actions):
        super(OutputPolicyNetwork, self).__init__()
        #in_features: actions*lstm_output_size + model free output size
        #self.fc = nn.Linear(512,256)    #TODO: change input size
        #self.policy_head = nn.Linear(512,nr_actions)
        #self.value_head = nn.Linear(512,1)

        self.fc = nn.Linear(input_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, nr_actions)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

    def forward(self,x):
        #x = F.relu(self.fc(x))
        #policy = self.policy_head(x)
        #value = self.value_head(x)
        x = F.relu(self.fc(x))
        policy = self.actor_linear(x)
        value = self.critic_linear(x)
        return (policy,value)
