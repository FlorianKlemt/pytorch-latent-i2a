import torch
from Models.MiniModel import MiniModel
from Models.model import ActorCritic

def load_policy(load_policy_model_dir = "trained_models/",
                policy_file = None,
                action_space = None,
                use_cuda = True,
                policy_name="MiniModel"):
    saved_state = torch.load('{0}{1}'.format(
        load_policy_model_dir, policy_file), map_location=lambda storage, loc: storage)

    if policy_name=="MiniModel":
        policy_model = MiniModel(num_inputs=4, action_space=action_space)
    elif policy_name=="OriginalModel":
        policy_model = ActorCritic(num_inputs=1, action_space=action_space)
    else:
        raise NotImplementedError("Model ",policy_name, " does not exist")
    policy_model.load_state_dict(saved_state)
    if use_cuda:
        policy_model.cuda()

    for param in policy_model.parameters():
        param.requires_grad = False

    policy_model.eval()
    return policy_model


def load_em_model(EMModel,
                  load_environment_model_dir = "trained_models/environment_models/",
                  environment_model_name = None,
                  num_inputs = None,
                  action_space = None,
                  use_cuda = True):

    saved_state = torch.load('{0}{1}'.format(
        load_environment_model_dir, environment_model_name), map_location=lambda storage, loc: storage)

    environment_model = EMModel(num_inputs=num_inputs,
                                num_actions=action_space,
                                use_cuda=use_cuda)
    environment_model.load_state_dict(saved_state)
    if use_cuda:
        environment_model.cuda()

    return environment_model