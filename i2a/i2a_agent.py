import torch

from i2a.output_policy_network import OutputPolicyNetwork


class I2A(torch.nn.Module):
    def __init__(self, model_free_network, model_based_network, action_space):
        super(I2A, self).__init__()

        self.model_free_network = model_free_network
        self.model_based_network = model_based_network

        output_policy_input_size = model_free_network.output_size() + model_based_network.output_size()
        output_policy_network = OutputPolicyNetwork(output_policy_input_size,
                                                    action_space)
        self.output_policy_network = output_policy_network
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
