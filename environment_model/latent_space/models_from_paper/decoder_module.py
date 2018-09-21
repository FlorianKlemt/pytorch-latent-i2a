import torch
import torch.nn as nn
from environment_model.latent_space.models_from_paper.depth2space import DepthToSpace
from environment_model.latent_space.models_from_paper.model_building_blocks import Flatten, ConvStack
from i2a.utils import get_linear_dims_after_conv


class DecoderModule(nn.Module):
    def __init__(self, input_shape, use_vae, reward_prediction_bits):
        super(DecoderModule, self).__init__()
        self.use_vae = use_vae
        reward_head_conv = nn.Conv2d(in_channels=input_shape[0], out_channels=24, kernel_size=3, stride=1)
        reward_head_linear_dims = get_linear_dims_after_conv([reward_head_conv], (input_shape[1], input_shape[2]))
        self.reward_head = nn.Sequential(
            reward_head_conv,
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=reward_head_linear_dims, out_features=reward_prediction_bits)
        )

        if self.use_vae:
            #state input channels + channels for z
            input_channels = input_shape[0] + input_shape[0]
        else:
            input_channels = input_shape[0]

        image_head_conv1_output_channels = 64
        image_head_d2s1_block_size = 2
        image_head_conv2_input_channels = image_head_conv1_output_channels/(pow(image_head_d2s1_block_size, 2))
        self.image_head = nn.Sequential(
            ConvStack(input_channels=input_channels, kernel_sizes=(1, 5, 3), output_channels=(32, 32, image_head_conv1_output_channels)),
            DepthToSpace(block_size=image_head_d2s1_block_size),
            ConvStack(input_channels=image_head_conv2_input_channels, kernel_sizes=(3, 3, 1), output_channels=(64, 64, 48)),
            DepthToSpace(block_size=4)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, state, z):
        reward_log_probs = self.reward_head(state)

        if self.use_vae and z is not None:
            concatenated = torch.cat((state, z), 1)
        else:
            concatenated = state

        image_log_probs = self.image_head(concatenated)
        image_log_probs = self.sigmoid(image_log_probs)
        return image_log_probs, reward_log_probs