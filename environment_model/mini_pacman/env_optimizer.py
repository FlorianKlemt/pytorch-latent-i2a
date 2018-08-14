import torch
import torch.nn as nn


class EnvMiniPacmanOptimizer():

    def __init__(self,
                 model,
                 reward_loss_coef,
                 lr, eps, weight_decay,
                 use_cuda):
        self.model = model
        if use_cuda == True:
            self.model.cuda()

        self.reward_loss_coef = reward_loss_coef
        self.loss_function_frame = torch.nn.BCELoss()
        self.loss_function_reward = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr = lr,
                                          eps = eps,
                                          weight_decay = weight_decay)

    def optimizer_step(self, sample):
        state, action, next_state_target, reward_target = sample
        self.optimizer.zero_grad()

        # Compute loss and gradient
        predicted_next_state, predicted_reward = self.model(state, action)

        next_frame_loss = self.loss_function_frame(predicted_next_state, next_state_target)
        next_reward_loss = self.loss_function_reward(predicted_reward, reward_target)

        # preform training step with both losses
        loss = next_reward_loss * self.reward_loss_coef + next_frame_loss
        loss.backward()

        self.optimizer.step()
        return (next_frame_loss, next_reward_loss), (predicted_next_state, predicted_reward)


