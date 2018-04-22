import torch.nn as nn

class EncoderModel(nn.Module):
    def __init__(self, num_inputs):
        super(EncoderModel, self).__init__()
        #self.encoder = nn.Sequential(
        #    nn.Linear(4*19*19, 128),    #1444 input size
        #    nn.ReLU(True),
        #    nn.Linear(128, 64))
        #self.decoder = nn.Sequential(
        #    nn.Linear(64, 128),
        #    nn.ReLU(True),
        #    nn.Linear(128, 4*19*19),
        #    nn.Tanh())

        self.encoder = nn.Sequential(
            nn.Linear(4 * 19 * 19, 1444))  # 1444 input size
        self.decoder = nn.Sequential(
            nn.Linear(1444, 4 * 19 * 19),
            nn.Tanh())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)

        x = x.view(1,4,19,19)
        return x