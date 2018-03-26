import torch

from I2A.ModelFreeNetwork import ModelFreeNetwork
from I2A.OuputPolicyNetwork import OutputPolicyNetwork
from I2A.ModelBasedNetwork import ModelBasedNetwork
from I2A.ImaginationCore import PongImaginationCore

from Environment_Model.environment_model import EMModel_used_for_Pong_I2A
from Environment_Model.load_imagination_core import load_imagination_core


class I2A(torch.nn.Module):
    def __init__(self, num_inputs, action_space, use_cuda):
        super(I2A, self).__init__()

        self.action_space = action_space
        self.input_channels = 1
        self.rollout_steps = 2
        self.number_lstm_cells = 256
        self.output_policy_input_size = 2048
        self.use_cuda = use_cuda

        self.output_policy_network = OutputPolicyNetwork(self.output_policy_input_size,
                                                         self.action_space)

        # model-free path
        self.model_free_network = ModelFreeNetwork(self.input_channels, 512)

        # model-based path
        self.model_based_network = ModelBasedNetwork(self.action_space,
                                                     self.input_channels,
                                                     self.number_lstm_cells,
                                                     self.rollout_steps,
                                                     self.use_cuda)





        self.train()


    def forward(self, input_state):
        self.model_based_network.repackage_lstm_hidden_variables()

        # model-free path
        model_free_result = self.model_free_network(input_state)

        # model-based path
        model_based_result = self.model_based_network(input_state)

        # aggregate model-free and model-based side and calculate policy and value
        aggregated_results = torch.cat((model_free_result, model_based_result), 1)
        policy, value = self.output_policy_network.forward(aggregated_results)
        return value, policy


    #for compatibility (leave this in for now)
    def repackage_lstm_hidden_variables(self):
        pass