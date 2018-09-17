import torch
from i2a.latent_space.models.latent_space_model_based_network import LatentSpaceModelBasedNetwork
from i2a.latent_space.models.latent_space_model_free_network import LatentSpaceModelFreeNetwork
from i2a.mini_pacman.models.output_policy_network import OutputPolicyNetwork


class I2ALatentSpace(torch.nn.Module):
    def __init__(self, obs_shape, action_space, imagination_core, rollout_steps, use_cuda):
        super(I2ALatentSpace, self).__init__()

        self.action_space = action_space
        self.rollout_steps = rollout_steps
        self.number_lstm_cells = 256
        self.model_free_output_size = 512
        self.model_based_output_size = self.number_lstm_cells * self.action_space
        self.output_policy_input_size = self.model_based_output_size + self.model_free_output_size
        self.use_cuda = use_cuda

        self.output_policy_network = OutputPolicyNetwork(self.output_policy_input_size,
                                                         self.action_space)

        # model-free path
        self.model_free_network = LatentSpaceModelFreeNetwork(obs_shape=obs_shape,
                                                             num_outputs=self.model_free_output_size)

        # model-based path
        self.model_based_network = LatentSpaceModelBasedNetwork(number_actions=self.action_space,
                                                                obs_shape=obs_shape,
                                                                imagination_core=imagination_core,
                                                                number_lstm_cells=self.number_lstm_cells,
                                                                rollout_steps=self.rollout_steps,
                                                                use_cuda=self.use_cuda)
        self.train()


    def forward(self, input_state):
        # model-free path
        model_free_result = self.model_free_network(input_state)

        # model-based path
        model_based_result = self.model_based_network(input_state)

        # aggregate model-free and model-based side and calculate policy and value
        aggregated_results = torch.cat((model_free_result, model_based_result), 1)
        policy, value = self.output_policy_network.forward(aggregated_results)
        return value, policy