#from bigger_model import Policy
from i2a.latent_space.models.i2a_agent import I2ALatentSpace
import torch
from i2a.latent_space.imagination_core.latent_space_imagination_core import LatentSpaceImaginationCore


def load_latent_space_environment_model(load_environment_model_path, latent_space_model, action_space, use_cuda):
    from environment_model.latent_space.models_from_paper.state_space_model import SSM
    environment_model = SSM(model_type=latent_space_model, observation_input_channels=3, state_input_channels=64,
                                 num_actions=action_space, use_cuda=use_cuda)

    print("Load environment model", load_environment_model_path)
    saved_state = torch.load(load_environment_model_path, map_location=lambda storage, loc: storage)
    environment_model.load_state_dict(saved_state)
    if use_cuda:
        environment_model.cuda()

    # the env_model does NOT require grads (require_grad=False) for now, to train jointly set to true
    for param in environment_model.parameters():
        param.requires_grad = False
    environment_model.eval()
    return environment_model

def build_latent_space_i2a_model(obs_shape,
                                 action_space,
                                 args):
    obs_shape_frame_stack = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    # Load Environment model
    environment_model_name = args.env_name + "_" + args.latent_space_model + ".dat"
    load_environment_model_path = 'trained_models/environment_models/' + environment_model_name
    environment_model = load_latent_space_environment_model(load_environment_model_path=load_environment_model_path,
                                                            latent_space_model=args.latent_space_model,
                                                            action_space=action_space.n,
                                                            use_cuda=args.cuda)
    # TODO policy input size is latent space size

    from i2a.rollout_policy import RolloutPolicy
    from i2a.latent_space.i2a_latent_space_actor_critic import I2ALatentSpaceActorCritic
    rollout_policy = RolloutPolicy(obs_shape=(64, 25, 20), action_space=action_space)
    for param in rollout_policy.parameters():
        param.requires_grad = True
    rollout_policy.train()
    if args.cuda:
        rollout_policy.cuda()

    imagination_core = LatentSpaceImaginationCore(env_model=environment_model,
                                                  rollout_policy=rollout_policy,
                                                  grey_scale=args.grey_scale, frame_stack=args.num_stack)

    i2a_model = I2ALatentSpaceActorCritic(policy=I2ALatentSpace(obs_shape=obs_shape_frame_stack,
                                                                action_space=action_space.n,
                                                                imagination_core=imagination_core,
                                                                rollout_steps=args.i2a_rollout_steps,
                                                                use_cuda=args.cuda),
                                          imagination_core=imagination_core,
                                          frame_stack=args.num_stack)

    return i2a_model