import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from I2A.EnvironmentModel.MiniPacmanEnvModel import MiniPacmanEnvModel
from I2A.load_utils import load_policy, load_em_model
from A2C_Models.A2C_PolicyWrapper import A2C_PolicyWrapper
from A2C_Models.I2A_MiniModel import I2A_MiniModel
import os

class ImaginationCore(nn.Module):
    def __init__(self, env_model=None, policy=None, use_cuda=False):
        super(ImaginationCore, self).__init__()
        self.env_model = env_model
        self.policy = policy
        if policy != None:
            self.num_actions = policy.num_actions
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    def forward(self, state, action):
        next_state, reward = self.env_model(state, action)

        x = torch.cat((state, next_state), 1)
        next_state = x[:, 1:]
        return next_state, reward

    def sample(self, input_state):
        #TODO: HOW THE FUCK IS THIS SUPPOSED TO BE WITH THE RETARDED ACT FUNCTION IN THIS REPO??
        #critic, actor = self.policy(state)
        value, action, action_log_probs, states = self.policy.act(input_state, None, None)
        #action = action.cpu().data[0][0]
        return action



class MiniPacmanImaginationCore(ImaginationCore):

    def __init__(self, num_inputs, action_space, em_model_reward_bins, use_cuda, require_grad):
        super(MiniPacmanImaginationCore, self).__init__(use_cuda=use_cuda)

        load_environment_model_dir = os.path.join(os.getcwd(), 'trained_models/environment_models/')
        self.env_model = load_em_model(EMModel=MiniPacmanEnvModel,
                                 load_environment_model_dir=load_environment_model_dir,
                                 environment_model_name="RegularMiniPacman_EnvModel_0.dat",
                                 num_inputs=num_inputs,
                                 action_space=action_space,
                                 reward_bins=em_model_reward_bins,
                                 use_cuda=use_cuda)

        if require_grad == False:
            for param in self.env_model.parameters():
                param.requires_grad = False
            self.env_model.eval()
        else:
            self.env_model.train()

        #load_policy_model_dir = os.path.join(os.getcwd(), 'trained_models/a2c/')
        #self.policy = load_policy(load_policy_model_dir=load_policy_model_dir,
        #                          policy_file="RegularMiniPacmanNoFrameskip-v0.pt",
        #                          action_space=action_space,
        #                          use_cuda=use_cuda,
        #                          policy_name="MiniModel")

        self.policy = A2C_PolicyWrapper(I2A_MiniModel(num_inputs=4, action_space=action_space, use_cuda=use_cuda))
        for param in self.policy.parameters():
            param.requires_grad = True
        self.policy.train()

        self.num_actions = action_space

        if use_cuda:
            self.env_model.cuda()
            self.policy.cuda()


