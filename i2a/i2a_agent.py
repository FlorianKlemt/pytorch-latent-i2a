import torch
from i2a.output_policy_network import OutputPolicyNetwork


class I2A(torch.nn.Module):
    def __init__(self, obs_shape, action_space, imagination_core, rollout_steps, use_cuda):
        super(I2A, self).__init__()

        self.obs_shape = obs_shape
        self.action_space = action_space
        self.imagination_core = imagination_core
        self.rollout_steps = rollout_steps
        self.number_lstm_cells = 256
        self.model_free_output_size = 512
        self.model_based_output_size = self.number_lstm_cells * self.action_space
        self.output_policy_input_size = self.model_based_output_size + self.model_free_output_size
        self.use_cuda = use_cuda

        self.output_policy_network = OutputPolicyNetwork(self.output_policy_input_size,
                                                         self.action_space)

        self._init_models()

        self.train()

    def _init_models(self):
        raise NotImplementedError('Do not directly use the I2A class. Instead use the subclasses of I2A [LatentSpaceI2A, ClassicalI2A]')


    def forward(self, input_state):
        # model-free path
        model_free_result = self.model_free_network(input_state)

        # model-based path
        model_based_result = self.model_based_network(input_state)

        # aggregate model-free and model-based side and calculate policy and value
        aggregated_results = torch.cat((model_free_result, model_based_result), 1)
        policy, value = self.output_policy_network.forward(aggregated_results)
        return value, policy


class LatentSpaceI2A(I2A):
    def __init__(self, obs_shape, encoding_shape, action_space, imagination_core, rollout_steps, frame_stack, use_cuda):
        self.encoding_shape = encoding_shape
        self.frame_stack = frame_stack
        super(LatentSpaceI2A, self).__init__(obs_shape, action_space, imagination_core, rollout_steps, use_cuda)

    def _init_models(self):
        from i2a.latent_space.models.latent_space_model_based_network import LatentSpaceModelBasedNetwork
        from i2a.latent_space.models.latent_space_model_free_network import LatentSpaceModelFreeNetwork
        self.model_free_network = LatentSpaceModelFreeNetwork(obs_shape=self.obs_shape,
                                                              num_outputs=self.model_free_output_size)

        self.model_based_network = LatentSpaceModelBasedNetwork(number_actions=self.action_space,
                                                                encoding_shape=self.encoding_shape,
                                                                imagination_core=self.imagination_core,
                                                                number_lstm_cells=self.number_lstm_cells,
                                                                rollout_steps=self.rollout_steps,
                                                                frame_stack=self.frame_stack,
                                                                use_cuda=self.use_cuda)

class ClassicI2A(I2A):
    def __init__(self, obs_shape, action_space, imagination_core, rollout_steps, use_cuda):
        super(ClassicI2A, self).__init__(obs_shape, action_space, imagination_core, rollout_steps, use_cuda)
    def _init_models(self):
        from i2a.mini_pacman.models.model_based_network import ModelBasedNetwork
        from i2a.mini_pacman.models.model_free_network import ModelFreeNetwork
        self.model_free_network = ModelFreeNetwork(obs_shape=self.obs_shape,
                                                   num_outputs=self.model_free_output_size)

        self.model_based_network = ModelBasedNetwork(number_actions=self.action_space,
                                                     obs_shape=self.obs_shape,
                                                     imagination_core=self.imagination_core,
                                                     number_lstm_cells=self.number_lstm_cells,
                                                     rollout_steps=self.rollout_steps,
                                                     use_cuda=self.use_cuda)

