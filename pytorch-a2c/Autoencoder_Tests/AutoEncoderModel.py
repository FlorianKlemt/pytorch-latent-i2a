import torch.nn as nn

class AutoEncoderModel(nn.Module):
    def __init__(self, num_inputs):
        super(AutoEncoderModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs*19*19, 128),    #1444 input size
            nn.ReLU(True),
            nn.Linear(128, 64))
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, num_inputs*19*19),
            nn.Tanh())

        #self.encoder = nn.Sequential(
        #    nn.Linear(19 * 19, 19*19))  # 1444 input size
        #self.decoder = nn.Sequential(
        #    nn.Linear(19*19, 19 * 19),
        #    nn.Tanh())

    def forward(self, x):
        x = x.view(-1)
        x = self.encoder(x)
        x = self.decoder(x)

        x = x.view(19,19)
        return x

    def encode(self, x):
        x = x.view(-1)
        return self.encoder(x)

    def decode(self, x):
        x = x.view(-1)
        x = self.decoder(x)
        return x.view(19,19)