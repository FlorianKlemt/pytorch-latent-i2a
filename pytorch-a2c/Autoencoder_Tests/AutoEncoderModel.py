import torch.nn as nn

class AutoEncoderModel(nn.Module):
    def __init__(self, num_inputs, input_size):
        super(AutoEncoderModel, self).__init__()
        self.input_size_x, self.input_size_y = input_size
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs*self.input_size_x*self.input_size_y, 128),    #1444 input size
            nn.ReLU(True),
            nn.Linear(128, 64))
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, num_inputs*self.input_size_x*self.input_size_y),
            )#nn.ReLU(True))

        #self.encoder = nn.Sequential(
        #    nn.Linear(19 * 19, 19*19))  # 1444 input size
        #self.decoder = nn.Sequential(
        #    nn.Linear(19*19, 19 * 19),
        #    nn.Tanh())

    def forward(self, x):
        x = x.view(-1)
        x = self.encoder(x)
        x = self.decoder(x)

        x = x.view(self.input_size_x,self.input_size_y)
        return x

    def encode(self, x):
        x = x.view(-1)
        return self.encoder(x)

    def decode(self, x):
        x = x.view(-1)
        x = self.decoder(x)
        return x.view(self.input_size_x, self.input_size_y)