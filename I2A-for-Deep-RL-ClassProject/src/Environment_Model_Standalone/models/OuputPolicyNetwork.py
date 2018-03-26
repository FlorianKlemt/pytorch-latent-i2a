import torch.nn as nn
import torch.nn.functional as F

class OutputPolicyNetwork(nn.Module):
    def __init__(self, nr_actions):
        super(OutputPolicyNetwork, self).__init__()
        #in_features: actions*lstm_output_size + model free output size
        self.fc = nn.Linear(TODO,256)
        self.policy_head = nn.Linear(256,nr_actions)
        self.value_head = nn.Linear(256,1)

    def forward(self,x):
        x = F.relu(self.fc(x))
        policy = F.relu(self.policy_head(x))
        value = F.relu(self.value_head(x))
        return (policy,value)