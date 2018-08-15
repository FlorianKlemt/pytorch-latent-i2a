import torch

def numerical_reward_to_bit_array(rewards, reward_prediction_bits, use_cuda):
    import math
    # one bit for sign, and one bit for 0
    reward_prediction_numerical_bits = reward_prediction_bits - 2
    max_representable_reward = int(math.pow(2, reward_prediction_numerical_bits) - 1)
    if use_cuda:
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
            number_str_format = '{0:0' + str(reward_prediction_numerical_bits) + 'b}'
            bits = [int(x) for x in list(number_str_format.format(abs(true_reward)))]
            for n in range(2, reward_prediction_bits):
                r_true[i, j, n] = bits[n - 2]
            # print(r_true[i,j], true_reward)
    return r_true