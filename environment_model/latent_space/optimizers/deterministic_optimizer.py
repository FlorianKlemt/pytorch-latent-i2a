import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli

from environment_model.latent_space.reward_to_bit import numerical_reward_to_bit_array

class DeterministicOptimizer():

    def __init__(self,
                 model,
                 reward_prediction_bits,
                 reward_loss_coef,
                 lr, eps, weight_decay,
                 use_cuda):
        self.model = model
        self.use_cuda = use_cuda
        if use_cuda:
            self.model.cuda()

        self.reward_prediction_bits = reward_prediction_bits
        self.reward_loss_coef = reward_loss_coef
        self.frame_criterion = nn.BCELoss()
        self.reward_criterion = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr = lr,
                                          eps = eps,
                                          weight_decay = weight_decay)



    def optimizer_step(self, sample):
        sample_observation_initial_context, sample_action_T, sample_next_observation_T, sample_reward_T = sample
        image_probs, reward_probs = self.model.forward_multiple(
            sample_observation_initial_context,
            sample_action_T)

        # reward loss
        true_reward = numerical_reward_to_bit_array(sample_reward_T, self.reward_prediction_bits, self.use_cuda)
        reward_loss = self.reward_criterion(reward_probs, true_reward)

        # image loss
        reconstruction_loss = self.frame_criterion(image_probs, sample_next_observation_T)

        loss = reconstruction_loss + self.reward_loss_coef * reward_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # The minimal cross entropy between the distributions p and q is the entropy of p
        # so if they are equal the loss is equal to the distribution of p
        true_entropy = Bernoulli(probs=sample_next_observation_T).entropy()
        normalized_frame_loss = reconstruction_loss - true_entropy.mean()
        return (normalized_frame_loss, reward_loss), (image_probs, reward_probs)


