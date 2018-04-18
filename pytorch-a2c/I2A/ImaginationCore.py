import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from I2A.EnvironmentModel.MiniPacmanEnvModel import MiniPacmanEnvModel
from I2A.load_utils import load_policy, load_em_model

class ImaginationCore(nn.Module):
    def __init__(self, env=None, policy=None, use_cuda=False):
        super(ImaginationCore, self).__init__()
        self.env = env
        self.policy = policy
        if policy != None:
            self.num_actions = policy.num_actions
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    def forward(self, state, action=None):
        if action == None:
            # TODO policy return order was wrong used -> there may be more errors like this!!
            critic, actor = self.policy(state)
            prob = F.softmax(actor, dim=1)
            action = prob.multinomial().data
        elif type(action) == int:
            pass
        else:
            raise IndexError("You passed an invalid action." +
                             "Expected to be < {}, but was {}".format(self.num_actions, action))

        np_action = np.zeros(self.num_actions)
        np_action[action] = 1
        action = Variable(torch.from_numpy(np_action), requires_grad=False).type(self.FloatTensor)

        state, reward = self.env(state, action)
        #reward = Variable(torch.from_numpy(np.ones(shape=(1)))).type(FloatTensor)

        return state, reward



class MiniPacmanImaginationCore(ImaginationCore):

    def __init__(self, num_inputs, use_cuda):
        super(MiniPacmanImaginationCore, self).__init__(use_cuda=use_cuda)

        self.env_model = load_em_model(EMModel=MiniPacmanEnvModel,
                                 load_environment_model_dir='/home/flo/Dokumente/I2A_GuidedResearch/pytorch-a2c/trained_models/environment_models/',
                                 environment_model_name="RegularMiniPacman_EnvModel_0.dat",
                                 num_inputs=num_inputs,
                                 action_space=5,
                                 use_cuda=use_cuda)

        for param in self.env_model.parameters():
            param.requires_grad = False
        self.env_model.eval()

        self.policy = load_policy(load_policy_model_dir="/home/flo/Dokumente/I2A_GuidedResearch/pytorch-a2c/trained_models/a2c/",
                                  policy_file="RegularMiniPacmanNoFrameskip-v0.pt",
                                  action_space=5,
                                  use_cuda=use_cuda,
                                  policy_name="MiniModel")

        for param in self.policy.parameters():
            param.requires_grad = False
        self.policy.eval()

        #self.num_actions = self.policy.num_actions

        if use_cuda:
            self.env_model.cuda()
            self.policy.cuda()


