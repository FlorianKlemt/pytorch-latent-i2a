import math
import torch.nn as nn
from environment_model.latent_space.models_from_paper.depth2space import SpaceToDepth
from environment_model.latent_space.models_from_paper.model_building_blocks import ConvStack


# All input dimension sizes (widht, height) need to be divisible by
# first_space_to_depth_block_size*second_space_to_depth_block_size (8 in our case)
class EncoderModule(nn.Module):
    def __init__(self, input_shape):
        super(EncoderModule, self).__init__()
        self.input_shape = input_shape
        first_space_to_depth_block_size = 4
        conv_stack1_input_channels = input_shape[0] * pow(first_space_to_depth_block_size, 2)
        conv_stack1_output_channels = 64
        second_space_to_depth_block_size = 2
        conv_stack2_input_channels = conv_stack1_output_channels * pow(second_space_to_depth_block_size, 2)
        self.output_channels = 64
        self.division_size = first_space_to_depth_block_size * second_space_to_depth_block_size
        assert (math.fmod(input_shape[1], self.division_size)==0 and math.fmod(input_shape[2], self.division_size)==0),\
            "Input Dimensions "+input_shape[1]+"x"+input_shape[2]+" need to be divisible by "+self.division_size
        self.encoder = nn.Sequential(
            SpaceToDepth(block_size=first_space_to_depth_block_size),
            ConvStack(input_channels=conv_stack1_input_channels, kernel_sizes=(3,5,3), output_channels=(16,16,conv_stack1_output_channels)),
            SpaceToDepth(block_size=second_space_to_depth_block_size),
            ConvStack(input_channels=conv_stack2_input_channels, kernel_sizes=(3,5,3), output_channels=(32,32,self.output_channels)),
            nn.ReLU()
        )

    def forward(self, observation):
        encoded = self.encoder(observation)
        return encoded

    def output_size(self):
        return (self.output_channels,int(self.input_shape[1]/self.division_size), int(self.input_shape[2]/self.division_size))

