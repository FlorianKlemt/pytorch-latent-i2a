import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class EnvEncoderModel(torch.nn.Module):
    def __init__(self, num_inputs, action_broadcast_size=10, use_cuda=True):
        super(EnvEncoderModel, self).__init__()
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.action_broadcast_size = action_broadcast_size

        self.linear1 = nn.Linear(64+self.action_broadcast_size,128)       #64+10 for latent space, 361 without
        self.linear2 = nn.Linear(128,64)

        self.train()

    def forward(self, inputs, action):
        broadcasted_reward = Variable(torch.zeros(self.action_broadcast_size))\
                                 .type(self.FloatTensor) + action
        x = torch.cat((inputs,broadcasted_reward),0)

        #x = inputs.view(-1, 19*19)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x