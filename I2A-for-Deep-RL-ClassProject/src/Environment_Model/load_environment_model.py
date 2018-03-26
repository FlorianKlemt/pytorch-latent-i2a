import torch

def load_em_model(EMModel,
                  load_environment_model_dir = "trained_models/environment_models/",
                  environment_model_name = "pong_em",
                  action_space = 6,
                  use_cuda = False):

    saved_state = torch.load('{0}{1}.dat'.format(
        load_environment_model_dir, environment_model_name), map_location=lambda storage, loc: storage)


    environment_model = EMModel(name=environment_model_name,
                                    num_input_actions=action_space,
                                    use_cuda=use_cuda)
    environment_model.load_state_dict(saved_state)
    if use_cuda:
        environment_model.cuda()

    return environment_model