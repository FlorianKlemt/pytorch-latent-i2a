import torch
from I2A.I2A_Models.ModelBasedNetwork import ModelBasedNetwork
from I2A.I2A_Models.ModelFreeNetwork import ModelFreeNetworkMiniPacman
from I2A.I2A_Models.OutputPolicyNetwork import OutputPolicyNetwork

#TODO: give I2A a parameter that defines the model to use, solve intriinsics here
#def ConfigModel():
#    if Model == "MiniPacman"

class I2A(torch.nn.Module):
    def __init__(self, num_inputs, action_space, use_cuda):
        super(I2A, self).__init__()

        self.action_space = action_space
        self.input_channels = 4
        self.rollout_steps = 2
        self.number_lstm_cells = 256
        self.model_free_output_size = 512
        self.model_based_output_size = self.number_lstm_cells * self.action_space
        self.output_policy_input_size = self.model_based_output_size + self.model_free_output_size
        self.use_cuda = use_cuda

        self.output_policy_network = OutputPolicyNetwork(self.output_policy_input_size,
                                                         self.action_space)

        # model-free path
        self.model_free_network = ModelFreeNetworkMiniPacman(self.input_channels)

        # model-based path
        self.model_based_network = ModelBasedNetwork(self.action_space,
                                                     self.input_channels,
                                                     self.number_lstm_cells,
                                                     self.rollout_steps,
                                                     self.use_cuda)
        self.train()


    def forward(self, input_state):
        #print(input_state.shape)
        self.model_based_network.repackage_lstm_hidden_variables()

        # model-free path
        model_free_result = self.model_free_network(input_state)

        # model-based path
        model_based_result = self.model_based_network(input_state)

        # aggregate model-free and model-based side and calculate policy and value
        aggregated_results = torch.cat((model_free_result, model_based_result), 1)
        policy, value = self.output_policy_network.forward(aggregated_results)
        return value, policy