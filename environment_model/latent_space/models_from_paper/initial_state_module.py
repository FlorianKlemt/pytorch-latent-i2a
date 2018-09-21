import torch
import torch.nn as nn
from environment_model.latent_space.models_from_paper.model_building_blocks import ConvStack


class InitialStateModule(nn.Module):
    def __init__(self):
        super(InitialStateModule, self).__init__()
        # encoded states of o_0, o_-1, o_-2 each 64 channels
        input_channels = 64 + 64 + 64
        self.conv_stack = ConvStack(input_channels=input_channels, kernel_sizes=(1,3,3), output_channels=(64,64,64))

    def forward(self, encoded_now, encoded_prev, encoded_2prev):
        concatenated = torch.cat((encoded_now, encoded_prev, encoded_2prev), 1)
        x = self.conv_stack(concatenated)
        return x
