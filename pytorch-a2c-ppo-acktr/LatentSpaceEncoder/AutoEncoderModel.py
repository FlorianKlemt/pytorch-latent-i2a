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

        x = x.view(self.num_inputs, self.input_size_x,self.input_size_y)
        return x

    def encode(self, x):
        x = x.view(-1)
        return self.encoder(x)

    def decode(self, x):
        x = x.view(-1)
        x = self.decoder(x)
        return x.view(self.num_inputs, self.input_size_x, self.input_size_y)



class Flatten(nn.Module):
    def forward(self,input):
        return input.view(input.size(0), -1)

class CNNAutoEncoderModel(AutoEncoderModel):
    def __init__(self, num_inputs, input_size, latent_space=64, hidden_space=128):
        super(CNNAutoEncoderModel, self).__init__()
        self.input_size_x, self.input_size_y = input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=3, padding=0),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(32 * 32 * 24, hidden_space),    #1444 input size
            nn.ReLU(True),
            nn.Linear(hidden_space, latent_space))
        self.decoder = nn.Sequential(
            nn.Linear(latent_space, hidden_space),
            nn.ReLU(True),
            nn.Linear(hidden_space, 1 * self.input_size_x * self.input_size_y),
            )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        x = self.encoder(x)
        return x.view(-1)

    def decode(self, x):
        x = self.decoder(x)
        return x.view(self.input_size_x, self.input_size_y)