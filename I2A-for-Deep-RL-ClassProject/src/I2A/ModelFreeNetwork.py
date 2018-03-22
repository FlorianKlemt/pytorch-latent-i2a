import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init

# see B.1: model free path uses identical network as the standard model-free baseline agent (withput the fc layers)
class ModelFreeNetworkSimple(nn.Module):
    def __init__(self, input_channels=1):
        super(ModelFreeNetworkSimple, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)

    def forward(self, x, reward):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        return x


class ModelFreeNetwork(torch.nn.Module):
    """
    This class implements the model free A3CLstm architecture implemented, the code is based on
    https://github.com/dgriff777/rl_a3c_pytorch. We also used the pretrained models if available.
    """

    def __init__(self, input_channels, num_outputs = 512):
        """
        The constructor need two integers, where the num_inputs describes the number of
        input channels and num_outputs is corresponding with the number of actions available in
        the specific game.
        :param num_inputs:
        :param num_outputs:
        """
        super(ModelFreeNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.fc = nn.Linear(2304, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


def load_model_A3C(path, num_inputs, action_space):
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
    model = ModelFreeNetwork(num_inputs, action_space)
    raw = torch.load(path)
    model.load_state_dict(raw)
    return model
