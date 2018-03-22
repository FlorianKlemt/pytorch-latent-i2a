import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from environment import atari_env

class ICPolicyNetwork(torch.nn.Module):
    """
    This class implements the model free A3CLstm architecture implemented, the code is based on
    https://github.com/dgriff777/rl_a3c_pytorch. We also used the pretrained models if available.
    """

    def __init__(self, num_inputs, num_actions, use_cuda):
        """
        The constructor need two integers, where the num_inputs describes the number of
        input channels and num_outputs is corresponding with the number of actions available in
        the specific game.
        :param num_inputs:
        :param num_outputs:
        """
        self.num_actions = num_actions

        super(ICPolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm = nn.LSTMCell(1024, 512)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        # USE VOLATILE!!! else big memory leak!!!!
        self.hx = Variable(torch.zeros(1, 512), requires_grad=False).type(FloatTensor)
        self.cx = Variable(torch.zeros(1, 512), requires_grad=False).type(FloatTensor)

        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)

        if use_cuda:
            self.cuda()

    def forward(self, inputs):
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))

        x = self.hx

        critic = self.critic_linear(x)
        actor = self.actor_linear(x)

        return actor, critic


    def repackage_lstm_hidden_variables(self):
        #print(self.cx.data.shape, self.hx.data.shape)
        self.hx = Variable(self.hx.data)
        self.cx = Variable(self.cx.data)


def load_model_A3C(path, num_inputs, action_space, use_cuda):
    """
    Example of usage:

    > import models.ModelFreeNetwork as mfn
    > import torch
    > from torch.autograd import Variable
    > mf_A3C = mfn.load_model_A3C('models/trained_models/A3C/Pong-v0.dat', 1, 6)
    > x = Variable(torch.FloatTensor([[[[1]*80]*80]]))
    > cx = Variable(torch.zeros(1, 512))
    > hx = Variable(torch.zeros(1, 512))
    > my_A3C( (x, (hx, cx)) ) # Forward Pass

    :param path:
    :param num_inputs:
    :param action_space:
    :return:
    """
    model = ICPolicyNetwork(num_inputs, action_space, use_cuda)
    raw = torch.load(path)
    model.load_state_dict(raw)
    return model


def load_model_pong(use_cuda):
    return load_model_A3C("trained_models/A3C/Pong-v0.dat", 1, 6, use_cuda=use_cuda)



import torch
import gym
from torch.autograd import Variable

def play_pong_with_pretrained_model(use_cuda):
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    my_A3C = load_model_pong(use_cuda)

    class ar:
        def __init__(self):
            self.skip_rate = 4

    env_config = {'crop1': 34, 'dimension2': 80, 'crop2': 34}
    args = ar()
    env = atari_env('Pong-v0', env_config, args)

    env.reset()
    for i_episode in range(20):
        observation = env.reset()

        for t in range(10000):
            env.render()
            #print(observation)

            observation = torch.from_numpy(observation).type(FloatTensor)
            # USE VOLATILE!!! else big memory leak!!!!
            observation = Variable(observation.unsqueeze(0), requires_grad=False, volatile=True)
            #observation = torch.unsqueeze(observation.permute(2, 0, 1), 0)
            #action = env.action_space.sample()
            action, critic = my_A3C((observation))

            action = np.argmax(action.data, 1)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break


#play_pong_with_pretrained_model(True)