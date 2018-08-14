import torch
import torch.nn as nn


class dSSM_DET_Optimizer():

    def __init__(self,
                 model,
                 reward_prediction_bits,
                 reward_loss_coef,
                 lr, eps, weight_decay,
                 use_cuda):
        self.model = model
        self.use_cuda = use_cuda
        if use_cuda == True:
            self.model.cuda()

        self.reward_prediction_bits = reward_prediction_bits
        self.reward_loss_coef = reward_loss_coef
        self.frame_criterion = torch.nn.BCELoss()
        self.reward_criterion = torch.nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr = lr,
                                          eps = eps,
                                          weight_decay = weight_decay)

    def numerical_reward_to_bit_array(self, rewards):
        import math
        reward_prediction_bits = self.reward_prediction_bits
        # one bit for sign, and one bit for 0
        reward_prediction_numerical_bits = reward_prediction_bits - 2
        max_representable_reward = int(math.pow(2, reward_prediction_numerical_bits) - 1)
        if self.use_cuda:
            r_true = torch.cuda.FloatTensor(rewards.shape[0], rewards.shape[1], reward_prediction_bits).fill_(0)
        else:
            r_true = torch.FloatTensor(rewards.shape[0], rewards.shape[1], reward_prediction_bits).fill_(0)
        for i in range(rewards.shape[0]):
            for j in range(rewards.shape[1]):
                true_reward = math.floor(rewards[i, j].item())  # they floor in the paper too
                if true_reward < -max_representable_reward:
                    print("True Reward too small to represent: ", true_reward, "<", -max_representable_reward)
                    true_reward = -max_representable_reward
                if true_reward > max_representable_reward:
                    print("True Reward too large to represent: ", true_reward, ">", max_representable_reward)
                    true_reward = max_representable_reward

                r_true[i, j, 0] = int(true_reward == 0)
                r_true[i, j, 1] = int(true_reward < 0)
                number_str_format = '{0:0'+str(reward_prediction_numerical_bits)+'b}'
                bits = [int(x) for x in list(number_str_format.format(abs(true_reward)))]
                for n in range(2, reward_prediction_bits):
                    r_true[i, j, n] = bits[n - 2]
                # print(r_true[i,j], true_reward)
        return r_true

    def optimizer_step(self, sample):
        sample_observation_initial_context, sample_action_T, sample_next_observation_T, sample_reward_T = sample
        image_probs, reward_probs = self.model.forward_multiple(
            sample_observation_initial_context,
            sample_action_T)

        # reward loss
        true_reward = self.numerical_reward_to_bit_array(sample_reward_T)
        reward_loss = self.reward_criterion(reward_probs, true_reward)

        # image loss
        reconstruction_loss = self.frame_criterion(image_probs, sample_next_observation_T)

        loss = reconstruction_loss + 1e-2 * reward_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # log and print infos
        #if self.loss_printer:
            # The minimal cross entropy between the distributions p and q is the entropy of p
            # so if they are equal the loss is equal to the distribution of p
        #    true_entropy = Bernoulli(probs=sample_next_observation_T).entropy()
        #   entropy_normalized_loss = reconstruction_loss - true_entropy.mean()
        #   return entropy_normalized_loss, reward_loss
        return (reconstruction_loss, reward_loss), (image_probs, reward_probs)


