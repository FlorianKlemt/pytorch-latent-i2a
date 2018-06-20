import torch.nn as nn
from LatentSpaceEncoder.models_from_paper.model_building_blocks import StateTransition, EncoderModule, DecoderModule

class dSSM_DET(nn.Module):
    def __init__(self, observation_input_channels, state_input_channels, num_actions, use_cuda):
        super(dSSM_DET, self).__init__()
        self.encoder = EncoderModule(input_channels=observation_input_channels)
        self.state_transition = StateTransition(state_input_channels=state_input_channels, num_actions=num_actions, use_stochastic=False, use_cuda=use_cuda)
        self.decoder = DecoderModule(state_input_channels=state_input_channels, use_vae=False)

    def forward(self, observation, action):
        encoding = self.encoder(observation)
        latent_state_prediction = self.state_transition(encoding, action, None) #no latent z for now
        image_log_probs, reward_log_probs = self.decoder(latent_state_prediction, None) #no latent z for now
        return image_log_probs, reward_log_probs
