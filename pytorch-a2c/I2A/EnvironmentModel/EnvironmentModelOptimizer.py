import torch
import torch.nn as nn
from collections import deque

class EnvironmentModelOptimizer():

    def __init__(self,
                 model,
                 lstm_backward_steps = 3,
                 use_cuda = True):

        self.use_cuda = use_cuda
        self.lstm_backward_steps = lstm_backward_steps
        # State Input
        self.model = model

        if self.use_cuda == True:
            self.model.cuda()

        self.loss_function_frame = nn.MSELoss()
        self.loss_function_reward = nn.MSELoss()
        #self.loss_function_reward = nn.CrossEntropyLoss()
        #self.loss_function_frame = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam
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