import torch
import torch.nn as nn
from collections import deque
from torch.autograd import Variable
import numpy as np
#from I2A.EnvironmentModel.minipacman_rgb_class_converter import MiniPacmanRGBToClassConverter

'''
def numpy_state_to_variable(state, use_cuda):
    state = Variable(torch.from_numpy(state).unsqueeze(0)).float()
    if use_cuda:
        state = state.cuda()
    return state


def numpy_reward_to_variable(reward, use_cuda):
    reward = Variable(torch.from_numpy(np.array([reward]))).float()
    if use_cuda:
        reward = reward.cuda()
    return reward
'''


class EnvironmentModelOptimizer():

    def __init__(self, model, args):
        self.model = model
        if args.cuda == True:
            self.model.cuda()

        self.loss_function_frame = nn.MSELoss()
        self.loss_function_reward = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr = args.lr,
                                          eps=args.eps,
                                          weight_decay=args.weight_decay)

    def optimizer_step(self, state, action, next_state_target, reward_target):
        self.optimizer.zero_grad()

        # Compute loss and gradient
        predicted_next_state, predicted_reward = self.model(state, action)

        next_frame_loss = self.loss_function_frame(predicted_next_state, next_state_target)
        next_reward_loss = self.loss_function_reward(predicted_reward, reward_target)

        # preform training step with both losses
        loss = next_reward_loss + next_frame_loss
        #loss.backward(retain_graph=True)
        loss.backward()

        self.optimizer.step()
        return (next_frame_loss, next_reward_loss), (predicted_next_state, predicted_reward)



class MiniPacmanEnvironmentModelOptimizer():

    def __init__(self, model, args):
        self.model = model
        if args.cuda == True:
            self.model.cuda()

        self.loss_function_reward = nn.MSELoss()
        #self.loss_function_reward = nn.CrossEntropyLoss()
        self.loss_function_state = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr = args.lr,
                                          eps=args.eps,
                                          weight_decay=args.weight_decay)

        from I2A.EnvironmentModel.minipacman_rgb_class_converter import MiniPacmanRGBToClassConverter
        self.rgb_to_class = MiniPacmanRGBToClassConverter()

    def optimizer_step(self,
                       state, action,
                       next_state_target, reward_target):
        self.optimizer.zero_grad()

        # Compute loss and gradient
        class_state = self.rgb_to_class.minipacman_rgb_to_class(state)
        class_next_state_target = self.rgb_to_class.minipacman_rgb_to_class(next_state_target)
        _, class_next_state_target = torch.max(class_next_state_target, 1)

        predicted_next_state, predicted_reward = self.model.forward_class(class_state, action)

        next_frame_loss = self.loss_function_state(predicted_next_state, class_next_state_target)
        next_reward_loss = self.loss_function_reward(predicted_reward, reward_target)

        # preform training step with both losses
        loss = next_reward_loss + next_frame_loss
        loss.backward()
        self.optimizer.step()

        predicted_next_state = self.rgb_to_class.minipacman_class_to_rgb(predicted_next_state)

        return (next_frame_loss, next_reward_loss), (predicted_next_state, predicted_reward)