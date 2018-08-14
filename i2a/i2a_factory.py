#from bigger_model import Policy
from a2c_models.a2c_policy_wrapper import A2C_PolicyWrapper, I2A_ActorCritic
from a2c_models.i2a_mini_model import I2A_MiniModel
from i2a.i2a_agent import I2A, I2ALatentSpace
import torch
from environment_model.mini_pacman.env_model import MiniPacmanEnvModel, CopyEnvModel, MiniPacmanEnvModelClassLabels
from i2a.load_utils import load_em_model
from i2a.imagination_core import ImaginationCore
from i2a.i2a_models.latent_space_imagination_core import LatentSpaceImaginationCore

def build_i2a_model(obs_shape,
                    action_space,
                    args,
                    em_model_reward_bins):
    color_prefix = 'grey_scale' if args.grey_scale else 'RGB'
    label_prefix = 'labels' if args.use_class_labels else 'NoLabels'
    frame_stack=args.num_stack
    i2a_rollout_steps=args.i2a_rollout_steps
    use_cuda=args.cuda
    environment_model_name=args.env_name + color_prefix + "_" + label_prefix + ".dat"
    use_copy_model=args.use_copy_model
    use_class_labels=args.use_class_labels



    input_channels = obs_shape[0]
    obs_shape_frame_stack = (obs_shape[0] * frame_stack, *obs_shape[1:])
    if use_copy_model:
        env_model = CopyEnvModel()
    else:
        # the env_model does NOT require grads (require_grad=False) for now, to train jointly set to true
        load_environment_model_dir = 'trained_models/environment_models/'
        if use_class_labels:
            obs_shape = (7, *obs_shape[1:])
            EMModel = MiniPacmanEnvModelClassLabels
        else:
            EMModel = MiniPacmanEnvModel

        env_model = load_em_model(EMModel=EMModel,
                                 load_environment_model_dir=load_environment_model_dir,
                                 environment_model_name=environment_model_name,
                                 obs_shape=obs_shape,
                                 action_space=action_space,
                                 reward_bins=em_model_reward_bins,
                                 use_cuda=use_cuda)

    for param in env_model.parameters():
        param.requires_grad = False
    env_model.eval()

    rollout_policy = A2C_PolicyWrapper(I2A_MiniModel(obs_shape=obs_shape_frame_stack, action_space=action_space, use_cuda=use_cuda))
    for param in rollout_policy.parameters():
        param.requires_grad = True
    rollout_policy.train()

    if use_cuda:
        env_model.cuda()
        rollout_policy.cuda()


    imagination_core = ImaginationCore(env_model=env_model, rollout_policy=rollout_policy,
                                       grey_scale=args.grey_scale, frame_stack=args.num_stack)

    i2a_model = I2A_ActorCritic(policy=I2A(obs_shape=obs_shape_frame_stack,
                                           action_space=action_space,
                                           imagination_core=imagination_core,
                                           rollout_steps=i2a_rollout_steps,
                                           use_cuda=args.cuda),
                                rollout_policy=rollout_policy)

    return i2a_model


def load_latent_space_environment_model(load_environment_model_path, latent_space_model, action_space, use_cuda):
    from LatentSpaceEncoder.models_from_paper.state_space_model import SSM
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
    from a2c_models.a2c_policy_wrapper import I2ALatentSpaceActorCritic
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