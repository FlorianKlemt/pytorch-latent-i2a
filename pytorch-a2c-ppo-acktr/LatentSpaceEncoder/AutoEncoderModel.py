from collections import OrderedDict
from I2A.utils import get_conv_output_dims, get_possible_sizes
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
    def __init__(self, num_inputs, latent_space, input_size):
        super(CNNAutoEncoderModel, self).__init__()
        self.input_size_x, self.input_size_y = input_size
        self.latent_space_dim = latent_space

        self.encoder_modules = OrderedDict([  # 1,3,3
            ('conv1', nn.Conv2d(in_channels=num_inputs, out_channels=16, kernel_size=9, stride=2, padding=0)),
            ('relu1', nn.ReLU(True)),
            # ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=None, padding=0)),
            ('conv2', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=2, padding=0)),
            ('relu2', nn.ReLU(True)),
            ('conv3', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=0)),
            ('relu3', nn.ReLU(True)),
            ('flatten', Flatten())])
        modified_convs = get_possible_sizes([self.encoder_modules['conv1'], self.encoder_modules['conv2'], self.encoder_modules['conv3']], input_size)
        print("Original Kernel Sizes: ",self.encoder_modules['conv1'].kernel_size,
                                        self.encoder_modules['conv2'].kernel_size,
                                        self.encoder_modules['conv3'].kernel_size,
              "New Kernel Sizes: ", modified_convs[0].kernel_size,
                                    modified_convs[1].kernel_size,
                                    modified_convs[2].kernel_size)
        self.encoder_modules['conv1'] = modified_convs[0]
        self.encoder_modules['conv2'] = modified_convs[1]
        self.encoder_modules['conv3'] = modified_convs[2]
        self.encoder = nn.Sequential(self.encoder_modules)

        conv_output_dims = get_conv_output_dims([self.encoder._modules['conv1'], self.encoder._modules['conv2'], self.encoder._modules['conv3']],
                                                input_size)
        conv_output_channels = self.encoder._modules['conv3'].out_channels
        self.latent_space_dim_in = reduce(lambda x, y: x * y, conv_output_dims) *conv_output_channels
        print("Latent in dim: ", self.latent_space_dim_in)
        self.encoder_linear = nn.Linear(self.latent_space_dim_in, self.latent_space_dim)
        self.decoder_linear = nn.Linear(self.latent_space_dim, self.latent_space_dim_in)
        self.decoder = nn.Sequential(
            Deflatten((conv_output_channels,) + conv_output_dims),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=self.encoder_modules['conv3'].kernel_size, stride=2, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=self.encoder_modules['conv2'].kernel_size, stride=2, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=num_inputs, kernel_size=self.encoder_modules['conv1'].kernel_size, stride=2, padding=0),
            )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = self.encoder_linear(x)
        return x

    def decode(self, x):
        x = self.decoder_linear(x)
        x = self.decoder(x)
        return x