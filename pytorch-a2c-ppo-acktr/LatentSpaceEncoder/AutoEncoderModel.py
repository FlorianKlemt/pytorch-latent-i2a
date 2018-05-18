from collections import OrderedDict
from I2A.utils import get_conv_output_dims
from functools import reduce

import torch.nn as nn

class AutoEncoderModel(nn.Module):
    def __init__(self):
        super(AutoEncoderModel, self).__init__()

    def forward(self):
        raise NotImplementedError('Should be implemented in subclass')

    def encode(self, x):
        raise NotImplementedError('Should be implemented in subclass')

    def decode(selfself, x):
        raise NotImplementedError('Should be implemented in subclass')


class LinearAutoEncoderModel(AutoEncoderModel):
    def __init__(self, num_inputs, input_size, latent_space=64, hidden_space=128):
        super(LinearAutoEncoderModel, self).__init__()
        self.num_inputs = num_inputs
        self.input_size_x, self.input_size_y = input_size
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs * self.input_size_x * self.input_size_y, hidden_space),    #1444 input size
            nn.ReLU(True),
            nn.Linear(hidden_space, latent_space))
        self.decoder = nn.Sequential(
            nn.Linear(latent_space, hidden_space),
            nn.ReLU(True),
            nn.Linear(hidden_space, num_inputs * self.input_size_x * self.input_size_y),
            )

    def forward(self, x):
        x = x.view(-1)
        x = self.encoder(x)
        x = self.decoder(x)

        x = x.view(1, self.num_inputs, self.input_size_x,self.input_size_y) #TODO: remove batchsize harcoding to 1
        return x

    def encode(self, x):
        x = x.view(-1)
        return self.encoder(x)

    def decode(self, x):
        x = x.view(-1)
        x = self.decoder(x)
        return x.view(1, self.num_inputs, self.input_size_x, self.input_size_y)



class Flatten(nn.Module):
    def forward(self,input):
        return input.view(input.size(0), -1)

class Deflatten(nn.Module):
    def __init__(self, shape):
        super(Deflatten, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view((-1,)+self.shape)

class CNNAutoEncoderModel(AutoEncoderModel):
    def __init__(self, num_inputs, input_size):
        super(CNNAutoEncoderModel, self).__init__()
        self.input_size_x, self.input_size_y = input_size
        self.encoder = nn.Sequential(OrderedDict([  #1,3,3
            ('conv1', nn.Conv2d(in_channels=num_inputs, out_channels=16, kernel_size=10, stride=4, padding=0)),
            ('relu1', nn.ReLU(True)),
            ('conv2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=10, stride=1, padding=0)),
            ('relu2', nn.ReLU(True)),
            ('conv3', nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=0)),
            ('relu3', nn.ReLU(True)),
            ('flatten', Flatten())]))

        conv_output_dims = get_conv_output_dims([self.encoder._modules['conv1'], self.encoder._modules['conv2'], self.encoder._modules['conv3']],
                                                input_size)
        conv_output_channels = self.encoder._modules['conv2'].out_channels
        self.latent_space_dim = reduce(lambda x, y: x * y, conv_output_dims) *conv_output_channels
        self.decoder = nn.Sequential(
            Deflatten((conv_output_channels,) + conv_output_dims),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=10, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=num_inputs, kernel_size=10, stride=4, padding=0),
            )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)