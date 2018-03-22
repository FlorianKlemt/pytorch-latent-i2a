import torch
import torch.nn as nn
import torch.nn.functional as F


# see B.1: model free path uses identical network as the standard model-free baseline agent (withput the fc layers)
class ModelFreeNetwork(nn.Module):
    def __init__(self, input_channels=1):
        super(ModelFreeNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)

    def forward(self, x, reward):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        return x


class ModelFreePathA3C(torch.nn.Module):
    """
    This class implements the model free A3CLstm architecture implemented, the code is based on
    https://github.com/dgriff777/rl_a3c_pytorch. We also used the pretrained models if available.
    """

    def __init__(self, num_inputs, num_outputs):
        """
        The constructor need two integers, where the num_inputs describes the number of
        input channels and num_outputs is corresponding with the number of actions available in
        the specific game.
        :param num_inputs:
        :param num_outputs:
        """
        super(ModelFreePathA3C, self).__init__()
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

        # We still have to initialize the unused critic_linear and actor_linear layers
        # for the model to be loaded properly.
        # UNUSED, BUT NEEDED >>>>>
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)
        # <<<<<<< UNUSED NEEDED

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))

        x = hx

        return x, (hx, cx)


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
    model = ModelFreePathA3C(num_inputs, action_space)
    raw = torch.load(path)
    model.load_state_dict(raw)
    return model
