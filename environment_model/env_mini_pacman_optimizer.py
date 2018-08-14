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



class EnvMiniPacmanLabelsOptimizer():

    def __init__(self,
                 model,
                 reward_loss_coef,
                 lr, eps, weight_decay,
                 use_cuda):
        self.model = model
        if use_cuda == True:
            self.model.cuda()

        self.reward_loss_coef = reward_loss_coef

        self.loss_function_reward = nn.MSELoss()
        self.loss_function_state = nn.CrossEntropyLoss()
        #self.loss_function_state = nn.HingeEmbeddingLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr = lr,
                                          eps= eps,
                                          weight_decay= weight_decay)

        from environment_model.minipacman_rgb_class_converter import MiniPacmanRGBToClassConverter
        self.rgb_to_class = MiniPacmanRGBToClassConverter(use_cuda=use_cuda)

    def optimizer_step(self, sample):
        state, action, next_state_target, reward_target = sample
        self.optimizer.zero_grad()

        # Compute loss and gradient
        class_state = self.rgb_to_class.minipacman_rgb_to_class(state)
        class_next_state_target = self.rgb_to_class.minipacman_rgb_to_class(next_state_target)
        _, class_next_state_target = torch.max(class_next_state_target, 1)

        predicted_next_state, predicted_reward = self.model.forward_class(class_state, action)

        next_frame_loss = self.loss_function_state(predicted_next_state, class_next_state_target)
        next_reward_loss = self.loss_function_reward(predicted_reward, reward_target)

        # preform training step with both losses
        loss = next_reward_loss * self.reward_loss_coef + next_frame_loss
        loss.backward()
        self.optimizer.step()

        predicted_next_state = self.rgb_to_class.minipacman_class_to_rgb(predicted_next_state)

        return (next_frame_loss, next_reward_loss), (predicted_next_state, predicted_reward)
