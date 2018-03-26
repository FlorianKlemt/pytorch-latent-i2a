import torch
from A3C_model import SmallA3Clstm, A3Clstm

def load_policy(load_policy_model_dir = "trained_models/",
                policy_file = "PongDeterministic-v4_21",
                action_space = 6,
                use_cuda = False,
                small_policy=True):
    saved_state = torch.load('{0}{1}.dat'.format(
        load_policy_model_dir, policy_file), map_location=lambda storage, loc: storage)

    if small_policy:
        policy_model = SmallA3Clstm(num_inputs=1, action_space=action_space, use_cuda=use_cuda)
    else:
        policy_model = A3Clstm(num_inputs=1, action_space=action_space, use_cuda=use_cuda)
    policy_model.load_state_dict(saved_state)
    if use_cuda:
        policy_model.cuda()

    for param in policy_model.parameters():
        param.requires_grad = False

    policy_model.eval()
    return policy_model