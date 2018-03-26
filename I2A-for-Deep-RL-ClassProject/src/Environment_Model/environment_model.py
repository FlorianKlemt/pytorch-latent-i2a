import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import deque
torch.manual_seed(42)


def init_weights(layer):
    """
    Initializes the weights of the given layer.
    Conv and Deconv layer will be initialized with a xavier initialization. Linear layers will
    be initialized uniform between -1 and 1
    :param layer:
    :return:
    """
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal(layer.weight.data)
        # xavier(m.bias.data)


class EMModel_Base(nn.Module):
    def __init__(self,
                 name,
                 use_cuda=False):
        super(EMModel_Base, self).__init__()
        self.name = name
        self.use_cuda = use_cuda


class EMModel_used_for_Pong_I2A(EMModel_Base):
    def __init__(self,
                 name,
                 env_state_dim=(1, 80, 80),
                 num_input_actions=9,
                 dropout=0.0,
                 use_cuda=False):
        """
        :param env_state_dim:
        :param num_input_actions:
        :param dropout:
        :param enable_tensorboard:
        """

        super(EMModel_used_for_Pong_I2A, self).__init__(name)

        self.name = name

        env_st_channels, env_st_height, env_st_width = env_state_dim

        # since we also add the tiled actions we get 1 + 2 actions for the default pong case
        input_channels = env_st_channels

        # State Input
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32,
                      kernel_size=8, stride=4, padding=0, bias=True),  # 19x19
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=5, stride=2, padding=0, bias=True),  # 8x8
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding=0, bias=True),  # 6x6
            nn.ReLU()
        )

        output_conv_features = 2304
        latent_space = 1024

        self.action_input = nn.Sequential(
            nn.Linear(in_features=num_input_actions,
                      out_features=output_conv_features
                      ),
            nn.ReLU(),
            nn.Linear(in_features=output_conv_features,
                      out_features=latent_space
                      ),
        )
        self.core_state_in = nn.Sequential(
            nn.Linear(in_features=output_conv_features,
                      out_features=latent_space
                      ),
            nn.ReLU(),
        )
        '''
        self.core_state_out = nn.Sequential(
            nn.Linear(in_features=1024,
                      out_features=output_conv_features
                      )
        )'''
        lstm_output_size = 1024
        self.lstm = nn.LSTMCell(latent_space, lstm_output_size)

        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.hx = Variable(torch.zeros(1, lstm_output_size)).type(FloatTensor)
        self.cx = Variable(torch.zeros(1, lstm_output_size)).type(FloatTensor)

        self.core_state_out = nn.Sequential(
            nn.Linear(in_features=lstm_output_size,
                      out_features=output_conv_features
                      )
        )
        # Output Heads
        # Next State Head
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=5, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=input_channels,
                               kernel_size=8, stride=4, padding=0, bias=True),
        )

        # NOTE: needs 1 as output channels otherwise the state_size grows with each rollout iteration
        self.rewardHead = nn.Sequential(
            nn.Linear(lstm_output_size, 256, bias=True),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, num_input_actions, bias=True)  # ,
            # nn.Softmax()
        )

        # Layer weight initalization
        for seq in self.children():
            for layer in seq.children():
                init_weights(layer)

    def forward(self, env_state_frame, env_input_action):  # input_action should be one_hot
        """

        :param env_state_frame: the current game state as a 1, 1, 80, 80 frame scaled to [0, 1]
        :param env_input_action: the current action that the agent takes
        :return: (env_state_frame_output, env_state_reward_output) -> new output state & new reward
        """

        # Convolution Layers
        state_conv = self.conv(env_state_frame)
        # Storing the output size of the conv layers to ensure the output layer will have the same size
        shape_out = state_conv.size()
        state_conv = state_conv.view(state_conv.size(0), -1)
        state_conv = self.core_state_in(state_conv)

        # Action input layers
        # TODO
        # broadcasted_action = Variable(torch.zeros(state_conv.data.shape))\
        #                         .type(self.FloatTensor) + env_input_action

        fc_action = self.action_input(env_input_action)
        if fc_action.size(0) == 512:
            fc_action = fc_action.view((1, 512))
            state_conv = state_conv.view((1, 512))

        # Core Layers

        # core = torch.cat((state_conv, fc_action), 1)
        core = state_conv * fc_action
        # state_deconv = self.core_state_out(core)
        self.hx, self.cx = self.lstm(core, (self.hx, self.cx))

        core = self.hx
        state_deconv = self.core_state_out(core)

        # Deconv Layers
        #   These layers are responsible for generating the next state
        state_deconv = state_deconv.view(shape_out)
        state_deconv = self.deconv(state_deconv)

        env_state_frame_output = state_deconv

        # Reward Layers
        #   These layers are responsible for calculating the reward of the
        env_state_reward_output = self.rewardHead(core)

        return env_state_frame_output, env_state_reward_output

    def repackage_lstm_hidden_variables(self):
        self.hx = Variable(self.hx.data)
        self.cx = Variable(self.cx.data)



class EMModel_LSTM_One_Reward(EMModel_Base):
    def __init__(self,
                 name,
                 env_state_dim=(1, 80, 80),
                 num_input_actions=9,
                 dropout=0.0,
                 use_cuda=False):
        """
        :param env_state_dim:
        :param num_input_actions:
        :param dropout:
        :param enable_tensorboard:
        """

        super(EMModel_LSTM_One_Reward, self).__init__(name=name,use_cuda=use_cuda)

        env_st_channels, env_st_height, env_st_width = env_state_dim

        # since we also add the tiled actions we get 1 + 2 actions for the default pong case
        input_channels = env_st_channels

        # State Input
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32,
                      kernel_size=8, stride=4, padding=0, bias=True), # 19x19
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=5, stride=2, padding=0, bias=True), # 8x8
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding=0, bias=True), # 6x6
            nn.ReLU()
        )

        output_conv_features = 2304
        latent_space = 1024

        self.action_input = nn.Sequential(
            nn.Linear(in_features=num_input_actions,
                      out_features=output_conv_features
                      ),
            nn.ReLU(),
            nn.Linear(in_features=output_conv_features,
                      out_features=latent_space
                      ),
        )
        self.core_state_in = nn.Sequential(
            nn.Linear(in_features=output_conv_features,
                      out_features=latent_space
                      ),
            nn.ReLU(),
        )
        '''
        self.core_state_out = nn.Sequential(
            nn.Linear(in_features=1024,
                      out_features=output_conv_features
                      )
        )'''
        lstm_output_size = 1024
        self.lstm = nn.LSTMCell(latent_space, lstm_output_size)

        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.hx = Variable(torch.zeros(1, lstm_output_size)).type(FloatTensor)
        self.cx = Variable(torch.zeros(1, lstm_output_size)).type(FloatTensor)

        self.core_state_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=lstm_output_size,
                      out_features=output_conv_features
                      ),
            nn.ReLU()
        )
        # Output Heads
        # Next State Head
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=5, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=input_channels,
                               kernel_size=8, stride=4, padding=0, bias=True),
        )

        # NOTE: needs 1 as output channels otherwise the state_size grows with each rollout iteration
        self.rewardHead = nn.Sequential(
            nn.Linear(lstm_output_size, 256, bias=True),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True)#,
            #nn.Softmax()
        )


        # Layer weight initalization
        for seq in self.children():
            for layer in seq.children():
                init_weights(layer)


    def forward(self, env_state_frame, env_input_action):  # input_action should be one_hot
        """

        :param env_state_frame: the current game state as a 1, 1, 80, 80 frame scaled to [0, 1]
        :param env_input_action: the current action that the agent takes
        :return: (env_state_frame_output, env_state_reward_output) -> new output state & new reward
        """

        # Convolution Layers
        state_conv = self.conv(env_state_frame)
        # Storing the output size of the conv layers to ensure the output layer will have the same size
        shape_out = state_conv.size()
        state_conv = state_conv.view(state_conv.size(0), -1)
        state_conv = self.core_state_in(state_conv)

        # Action input layers
        # TODO
        #broadcasted_action = Variable(torch.zeros(state_conv.data.shape))\
        #                         .type(self.FloatTensor) + env_input_action

        fc_action = self.action_input(env_input_action)
        if fc_action.size(0) == 512:
            fc_action = fc_action.view((1, 512))
            state_conv = state_conv.view((1, 512))

        # Core Layers

        #core = torch.cat((state_conv, fc_action), 1)
        core = state_conv * fc_action
        #state_deconv = self.core_state_out(core)
        self.hx, self.cx = self.lstm(core, (self.hx, self.cx))

        core = self.hx
        state_deconv = self.core_state_out(core)

        # Deconv Layers
        #   These layers are responsible for generating the next state
        state_deconv = state_deconv.view(shape_out)
        state_deconv = self.deconv(state_deconv)

        env_state_frame_output = state_deconv

        # Reward Layers
        #   These layers are responsible for calculating the reward of the
        env_state_reward_output = self.rewardHead(core)

        return env_state_frame_output, env_state_reward_output


    def repackage_lstm_hidden_variables(self):
        self.hx = Variable(self.hx.data)
        self.cx = Variable(self.cx.data)




class PongEM_Big_Model(EMModel_Base):

    def __init__(self, name, env_state_dim=(1, 80, 80), num_input_actions=9, dropout=0.0, enable_tensorboard=True):
        """
        :param env_state_dim:
        :param num_input_actions:
        :param dropout:
        :param enable_tensorboard:
        """

        super(PongEM_Big_Model, self).__init__(name=name,use_cuda=use_cuda)

        self.name = name

        env_st_channels, env_st_height, env_st_width = env_state_dim

        # since we also add the tiled actions we get 1 + 2 actions for the default pong case
        input_channels = env_st_channels

        # State Input
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=64,
                      kernel_size=8,
                      stride=2,
                      padding=1,
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=6,
                      stride=2,
                      padding=1,
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=4,
                      stride=2,
                      padding=0,
                      bias=True),
            nn.ReLU()
        )

        self.action_input = nn.Sequential(
            nn.Linear(in_features=num_input_actions,
                      out_features=1024
                      ),
            nn.ReLU(),
            nn.Linear(in_features=1024,
                      out_features=2048
                      ),
        )

        self.core_state_in = nn.Sequential(
            nn.Linear(in_features=1152,
                      out_features=2048
                      ),
            nn.ReLU(),
        )

        self.core_state_out = nn.Sequential(
            nn.Linear(in_features=2048,
                      out_features=1152
                      )
        )

        # Output Heads

        # Next State Head
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=0,
                               bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=128,
                               kernel_size=6,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=128,
                               kernel_size=6,
                               stride=2,
                               padding=1,
                               bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=1,
                               kernel_size=8,
                               stride=2,
                               padding=1,
                               bias=True)
        )

        # NOTE: needs 1 as output channels otherwise the state_size grows with each rollout iteration
        self.rewardHead = nn.Sequential(
            nn.Linear(2048, 512, bias=True),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(512, 128, bias=True),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, num_input_actions, bias=True),
            nn.Softmax()
        )


    def forward(self, env_state_frame, env_input_action):  # input_action should be one_hot
        """

        :param env_state_frame: the current game state as a 1, 1, 80, 80 frame scaled to [0, 1]
        :param env_input_action: the current action that the agent takes
        :return: (env_state_frame_output, env_state_reward_output) -> new output state & new reward
        """

        # Convolution Layers
        state_conv = self.conv(env_state_frame)
        # Storing the output size of the conv layers to ensure the output layer will have the same size
        shape_out = state_conv.size()
        state_conv = state_conv.view(state_conv.size(0), -1)
        state_conv = self.core_state_in(state_conv)

        # Action input layers
        fc_action = self.action_input(env_input_action)
        if fc_action.size(0) == 2048:
            fc_action = fc_action.view((1, 2048))
            state_conv = state_conv.view((1, 2048))
        # Core Layers
        core = state_conv
        core = state_conv * fc_action
        state_deconv = self.core_state_out(core)

        # Deconv Layers
        #   These layers are responsible for generating the next state
        state_deconv = state_deconv.view(shape_out)
        state_deconv = self.deconv(state_deconv)

        # scale the output (-> predicted frame) to [0, 1]
        out_min, out_max = state_deconv.min(), state_deconv.max()
        env_state_frame_output = (state_deconv - out_min) / (out_max - out_min)

        # Reward Layers
        #   These layers are responsible for calculating the reward of the
        env_state_reward_output = self.rewardHead(core)

        return env_state_frame_output, env_state_reward_output





class EMModel_Same_Kernel_Size(EMModel_Base):
    def __init__(self,
                 name,
                 env_state_dim=(1, 80, 80),
                 num_input_actions=9,
                 dropout=0.0,
                 use_cuda=False):

        super(EMModel_Same_Kernel_Size, self).__init__(name=name,use_cuda=use_cuda)

        self.name = name

        env_st_channels, env_st_height, env_st_width = env_state_dim

        # since we also add the tiled actions we get 1 + 2 actions for the default pong case
        input_channels = env_st_channels

        # State Input
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64,
                      kernel_size=6, stride=2, padding=0, bias=True), # ((80 - 6) / 2) + 1 = 38 x38
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=6, stride=2, padding=1, bias=True), # ((38 - 6 + 2) / 2) + 1 = 18x18
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=6, stride=2, padding=0, bias=True), # ((18 - 6) / 2) + 1 = 7x7
            nn.ReLU()
        )

        output_conv_features = 3136

        self.core_features_in = 512
        self.latent_space = 512

        self.action_input = nn.Sequential(
            nn.Linear(in_features=num_input_actions,
                      out_features=256
                      ),
            nn.ReLU(),
            nn.Linear(in_features=256,
                      out_features=self.core_features_in
                      ),
            nn.ReLU()
        )

        self.core_state_in = nn.Sequential(
            nn.Linear(in_features=output_conv_features,
                      out_features=self.core_features_in
                      ),
            nn.ReLU()
        )

        #self.core_action_state_out = nn.Sequential(
        #    nn.Linear(in_features=self.core_features_in,
        #              out_features=self.latent_space
        #              ),
        #    nn.ReLU()
        #)

        lstm_output_size = 1024
        self.lstm = nn.LSTMCell(self.latent_space, lstm_output_size)
        self.relu_lstm = nn.ReLU()

        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.hx = Variable(torch.zeros(1, lstm_output_size)).type(FloatTensor)
        self.cx = Variable(torch.zeros(1, lstm_output_size)).type(FloatTensor)

        self.core_state_out = nn.Sequential(
            nn.Linear(in_features=lstm_output_size,
                      out_features=output_conv_features
                      ),
            nn.ReLU()
        )
        # Output Heads
        # Next State Head
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64,
                               kernel_size=6, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=6, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=input_channels,
                               kernel_size=6, stride=2, padding=0, bias=True),
        )

        # NOTE: needs 1 as output channels otherwise the state_size grows with each rollout iteration
        self.rewardHead = nn.Sequential(
            nn.Linear(1024, 256, bias=True),
            nn.ReLU(),
            #nn.Linear(256, 3, bias=True),# one hot encoding for one reward (-1, 0, 1)
            #nn.Softmax()
            nn.Linear(256, 1, bias=True) # predict one reward for (state, action) -> r
        )


        # Layer weight initalization
        for seq in self.children():
            for layer in seq.children():
                init_weights(layer)


    def forward(self, env_state_frame, env_input_action):  # input_action should be one_hot
        """

        :param env_state_frame: the current game state as a 1, 1, 80, 80 frame scaled to [0, 1]
        :param env_input_action: the current action that the agent takes
        :return: (env_state_frame_output, env_state_reward_output) -> new output state & new reward
        """

        # Convolution Layers
        state_conv = self.conv(env_state_frame)

        # Storing the output size of the conv layers to ensure the output layer will have the same size
        shape_out = state_conv.size()
        state_conv = state_conv.view(state_conv.size(0), -1)
        state_conv = self.core_state_in(state_conv)

        # Action input layers
        fc_action = self.action_input(env_input_action)
        fc_action = fc_action.view((1, -1))
        state_conv = state_conv.view((1, -1))
        #if fc_action.size(0) == 512:
        #    fc_action = fc_action.view((1, 512))
        #    state_conv = state_conv.view((1, 512))

        # Core Layers
        core = state_conv * fc_action

        #core = self.core_action_state_out(core)

        #state_deconv = self.core_state_out(core)
        self.hx, self.cx = self.lstm(core, (self.hx, self.cx))
        core = self.hx
        core = self.relu_lstm(core)

        # Deconv Layers
        state_deconv = self.core_state_out(core)

        #   These layers are responsible for generating the next state
        state_deconv = state_deconv.view(shape_out)
        state_deconv = self.deconv(state_deconv)

        env_state_frame_output = state_deconv

        # Reward Layers
        #   These layers are responsible for calculating the reward of the
        core = core.view((1, -1))
        env_state_reward_output = self.rewardHead(core)

        return env_state_frame_output, env_state_reward_output


    def repackage_lstm_hidden_variables(self):
        self.hx = Variable(self.hx.data)
        self.cx = Variable(self.cx.data)




class EnvironmentModelOptimizer():

    def __init__(self,
                 model,
                 lstm_backward_steps = 3,
                 use_cuda = False):

        self.use_cuda = use_cuda
        self.lstm_backward_steps = lstm_backward_steps
        # State Input
        self.model = model

        if self.use_cuda == True:
            self.model.cuda()

        self.loss_function_frame = nn.MSELoss()
        #self.loss_function_reward = nn.CrossEntropyLoss()
        #self.loss_function_frame = nn.CrossEntropyLoss()
        self.loss_function_reward = nn.MSELoss()

        self.optimizer = torch.optim.Adam
        #self.optimizer = torch.optim.RMSprop
        self.last_states_actions = deque(maxlen=self.lstm_backward_steps)

    def set_optimizer(self, optimizer_args_adam = {"lr": 1e-4,
                                                   "betas": (0.9, 0.999),
                                                   "eps": 1e-8,
                                                   "weight_decay": 0.00001}):
        # initialize optimizer
        self.optimizer = self.optimizer(self.model.parameters(), **optimizer_args_adam)

    def optimizer_step(self,
                       env_state_frame, env_action,
                       env_state_frame_target, env_reward_target):
        """
        Make a single gradient update.
        """
        self.optimizer.zero_grad()
        _, executed_action_index = env_action.max(0)

        self.last_states_actions.append((env_state_frame, env_action))

        # Compute loss and gradient
        for state, action in self.last_states_actions:
            next_frame, next_reward = self.model(state, action)


        next_frame_loss = self.loss_function_frame(next_frame, env_state_frame_target)
        next_reward_loss = self.loss_function_reward(next_reward, env_reward_target)

        #print(next_reward_loss.data)
        # preform training step with both losses
        #loss = next_reward_loss + next_reward_loss
        #self.optimizer.zero_grad()
        #torch.autograd.backward(loss, retain_graph=True)
        losses = [next_frame_loss, next_reward_loss]
        grad_seq = [losses[0].data.new(1).fill_(1) for _ in range(len(losses))]
        torch.autograd.backward(losses, grad_seq, retain_graph=True)
        #torch.autograd.backward(losses, retain_graph=True)

        self.optimizer.step()

        self.model.repackage_lstm_hidden_variables()

        return (next_frame_loss, next_reward_loss), (next_frame, next_reward)



