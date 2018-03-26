import os
from I2A.ImaginationCore import ImaginationCore
from Environment_Model.load_small_a3c_policy_model import load_policy
from Environment_Model.load_environment_model import load_em_model


def load_imagination_core(action_space=6,
                          policy_model = "PongDeterministic-v4_21",
                          load_policy_model_dir = "trained_models/",
                          EMModel = None,
                          environment_model_name = "pong_em",
                          load_environment_model_dir="trained_models/environment_models/",
                          root_path="",
                          use_cuda=False):
    load_policy_model_dir = os.path.join(root_path, load_policy_model_dir)
    load_environment_model_dir = os.path.join(root_path, load_environment_model_dir)

    em_model = load_em_model(EMModel=EMModel,
                             load_environment_model_dir= load_environment_model_dir,
                             environment_model_name= environment_model_name,
                             action_space=action_space,
                             use_cuda=use_cuda)
    policy = load_policy(load_policy_model_dir,
                         policy_model,
                         action_space=action_space,
                         use_cuda=use_cuda)

    if use_cuda:
        em_model.cuda()
    return ImaginationCore(em_model, policy, use_cuda)

