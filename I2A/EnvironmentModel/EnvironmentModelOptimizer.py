import torch
import torch.nn as nn
from collections import deque
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

        self.reward_loss_coef = args.reward_loss_coef

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
        loss = next_reward_loss * self.reward_loss_coef + next_frame_loss
        #loss.backward(retain_graph=True)
        loss.backward()

        self.optimizer.step()
        return (next_frame_loss, next_reward_loss), (predicted_next_state, predicted_reward)


    def rollout_optimizer_step(self, rollout_batch):
        self.optimizer.zero_grad()

        next_frame_loss, next_reward_loss = 0, 0
        '''state_batch, _, _, _ = [torch.cat(a) for a in zip(*[rollout[0] for rollout in rollout_batch])]
        for i in range(1, len(rollout_batch[0])):
            single_rollout_step_batch = [rollout[i] for rollout in rollout_batch]
            #state_batch, action_batch, next_state_batch, reward_batch = [torch.cat(a) for a in zip(*single_rollout_step_batch)]
            _ , action_batch, next_state_batch, reward_batch = [torch.cat(a) for a in zip(*single_rollout_step_batch)]

            # Compute loss and gradient
            predicted_next_state, predicted_reward = self.model(state_batch, action_batch)

            next_frame_loss += self.loss_function_frame(predicted_next_state, next_state_batch)
            next_reward_loss += self.loss_function_reward(predicted_reward, reward_batch)

            state_batch = predicted_next_state #??'''

        for i in range(len(rollout_batch[0])):
            single_rollout_step_batch = [rollout[i] for rollout in rollout_batch]
            state_batch, action_batch, next_state_batch, reward_batch = [torch.cat(a) for a in zip(*single_rollout_step_batch)]

            # Compute loss and gradient
            predicted_next_state, predicted_reward = self.model(state_batch, action_batch)

            next_frame_loss += self.loss_function_frame(predicted_next_state, next_state_batch)
            next_reward_loss += self.loss_function_reward(predicted_reward, reward_batch)


        # preform training step with both losses
        loss = next_reward_loss * self.reward_loss_coef + next_frame_loss
        loss.backward()

        self.optimizer.step()
        return (next_frame_loss, next_reward_loss), (predicted_next_state, predicted_reward)



class MiniPacmanEnvironmentModelOptimizer():

    def __init__(self, model, args):
        self.model = model
        if args.cuda == True:
            self.model.cuda()

        self.reward_loss_coef = args.reward_loss_coef

        self.loss_function_reward = nn.MSELoss()
        self.loss_function_state = nn.CrossEntropyLoss()
        #self.loss_function_state = nn.HingeEmbeddingLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr = args.lr,
                                          eps=args.eps,
                                          weight_decay=args.weight_decay)

        from I2A.EnvironmentModel.minipacman_rgb_class_converter import MiniPacmanRGBToClassConverter
        self.rgb_to_class = MiniPacmanRGBToClassConverter(use_cuda=args.cuda)

    def optimizer_step(self,
                       state, action,
                       next_state_target, reward_target):
        self.optimizer.zero_grad()

        # Compute loss and gradient
        class_state = self.rgb_to_class.minipacman_rgb_to_class(state)
        class_next_state_target = self.rgb_to_class.minipacman_rgb_to_class(next_state_target)
        _, class_next_state_target = torch.max(class_next_state_target, 1)

        #hinge_class_next_state_target = torch.ones(class_state.data.shape).type(torch.cuda.FloatTensor)
        #hinge_class_next_state_target *= -1
        #hinge_class_next_state_target = hinge_class_next_state_target.scatter_(1, torch.unsqueeze(class_next_state_target.data, 1), 1)

        predicted_next_state, predicted_reward = self.model.forward_class(class_state, action)

        next_frame_loss = self.loss_function_state(predicted_next_state, class_next_state_target)
        #next_frame_loss = self.loss_function_state(predicted_next_state.data, hinge_class_next_state_target)
        next_reward_loss = self.loss_function_reward(predicted_reward, reward_target)

        # preform training step with both losses
        loss = next_reward_loss * self.reward_loss_coef + next_frame_loss
        loss.backward()
        self.optimizer.step()

        predicted_next_state = self.rgb_to_class.minipacman_class_to_rgb(predicted_next_state)

        return (next_frame_loss, next_reward_loss), (predicted_next_state, predicted_reward)

    def rollout_optimizer_step(self, rollout_batch):
        self.optimizer.zero_grad()

        next_frame_loss, next_reward_loss = 0, 0
        for i in range(len(rollout_batch[0])):
            single_rollout_step_batch = [rollout[i] for rollout in rollout_batch]
            state_batch, action_batch, next_state_batch, reward_batch = [torch.cat(a) for a in zip(*single_rollout_step_batch)]

            class_state = self.rgb_to_class.minipacman_rgb_to_class(state_batch)
            class_next_state_target = self.rgb_to_class.minipacman_rgb_to_class(next_state_batch)
            _, class_next_state_target = torch.max(class_next_state_target, 1)

            # Compute loss and gradient
            predicted_next_state, predicted_reward = self.model.forward_class(class_state, action_batch)

            next_frame_loss += self.loss_function_state(predicted_next_state, class_next_state_target)
            next_reward_loss += self.loss_function_reward(predicted_reward, reward_batch)


        # preform training step with both losses
        loss = next_reward_loss * self.reward_loss_coef + next_frame_loss
        loss.backward()

        self.optimizer.step()
        return (next_frame_loss, next_reward_loss), (predicted_next_state, predicted_reward)