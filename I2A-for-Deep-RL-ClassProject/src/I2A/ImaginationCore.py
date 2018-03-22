import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

import torch.nn.functional as F
from I2A.IC_PolicyNetwork import load_model_A3C

from Environment_Model.load_environment_model import load_em_model
from Environment_Model.environment_model import EMModel_used_for_Pong_I2A

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

    def repackage_lstm_hidden_variables(self):
        self.policy.repackage_lstm_hidden_variables()



class PongImaginationCore(ImaginationCore):

    def __init__(self, use_cuda):
        super(PongImaginationCore, self).__init__(use_cuda=use_cuda)

        self.env = load_em_model(EMModel=EMModel_used_for_Pong_I2A,
                                 load_environment_model_dir='trained_models/environment_models/',
                                 environment_model_name="small_pong_em_lstm",
                                 action_space=6,
                                 use_cuda=use_cuda)

        for param in self.env.parameters():
            param.requires_grad = False
        self.env.eval()

        self.policy = load_model_A3C("trained_models/A3C/Pong-v0.dat",
                                     num_inputs=1, action_space= 6, use_cuda=use_cuda)

        for param in self.policy.parameters():
            param.requires_grad = False
        self.policy.eval()

        self.num_actions = self.policy.num_actions

        if use_cuda:
            self.env.cuda()
            self.policy.cuda()


