import torch.nn as nn
from LatentSpaceEncoder.models_from_paper.model_building_blocks import StateTransition, EncoderModule, DecoderModule, PriorModule, PosteriorModule

class sSSM(nn.Module):
    def __init__(self, observation_input_channels, state_input_channels, num_actions, use_cuda):
        super(sSSM, self).__init__()
        self.encoder = EncoderModule(input_channels=observation_input_channels)
        self.state_transition = StateTransition(state_input_channels=state_input_channels, num_actions=num_actions, use_stochastic=True, use_cuda=use_cuda)
        self.decoder = DecoderModule(state_input_channels=state_input_channels, use_vae=True)

        self.prior_z = PriorModule(state_input_channels=state_input_channels, num_actions=num_actions, use_cuda=use_cuda)

    def forward(self, observation, action):
        #TODO: second forward function which takes state and action
        state = self.encoder(observation)

        z = self.prior_z(state, action)

        next_state_prediction = self.state_transition(state, action, z)
        image_log_probs, reward_log_probs = self.decoder(next_state_prediction, z)
        return image_log_probs, reward_log_probs
