import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli

class EnvMiniPacmanOptimizer():

    def __init__(self,
                 model,
                 reward_loss_coef,
                 lr, eps, weight_decay,
                 use_cuda):
        self.model = model
        if use_cuda:
            self.model.cuda()

        self.reward_loss_coef = reward_loss_coef
        self.loss_function_frame = nn.BCELoss()
        self.loss_function_reward = nn.L1Loss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr = lr,
                                          eps = eps,
                                          weight_decay = weight_decay)

    def optimizer_step(self, sample):
        state, action, next_state_target, reward_target = sample
        self.optimizer.zero_grad()

        # Compute loss and gradient
        predicted_next_state, predicted_reward = self.model(state, action)

        # image loss
        reconstruction_loss = self.loss_function_frame(predicted_next_state, next_state_target)

        # reward loss
        reward_loss = self.loss_function_reward(predicted_reward, reward_target)

        # preform training step with both losses
        loss = reconstruction_loss + reward_loss * self.reward_loss_coef
        loss.backward()

        self.optimizer.step()

        # The minimal cross entropy between the distributions p and q is the entropy of p
        # so if they are equal the loss is equal to the distribution of p
        true_entropy = Bernoulli(probs=next_state_target).entropy()
        normalized_frame_loss = reconstruction_loss - true_entropy.mean()
        return (normalized_frame_loss, reward_loss), (predicted_next_state, predicted_reward)


