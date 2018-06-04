import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class EnvEncoderModel(torch.nn.Module):
    def __init__(self, num_inputs, latent_space=64, encoder_space=128, action_broadcast_size=10, use_cuda=True):
        super(EnvEncoderModel, self).__init__()
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.action_broadcast_size = action_broadcast_size

        self.linear1 = nn.Linear(latent_space+self.action_broadcast_size,encoder_space)       #64+10 for latent space, 361 without

        self.img_head = nn.Linear(encoder_space,latent_space)
        self.reward_head = nn.Linear(encoder_space, 1)

        self.train()

    def forward(self, inputs, action):
        #broadcasted_action= action.repeat(inputs.data.shape[0],self.action_broadcast_size)
        #x = torch.cat((inputs,broadcasted_action),1)

        # TODO: assuming batch size 1
        inputs = inputs.unsqueeze(0)    #TODO: fml, do this line if and only if is it minipacman
        broadcasted_action = action.repeat(1, self.action_broadcast_size).float()
        x = torch.cat((inputs,broadcasted_action),1)

        x = F.relu(self.linear1(x))
        img_out = self.img_head(x)
        reward_out = self.reward_head(x)

        return img_out, reward_out