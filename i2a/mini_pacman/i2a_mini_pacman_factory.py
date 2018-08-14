#from bigger_model import Policy
from a2c_models.a2c_policy_wrapper import A2C_PolicyWrapper
from i2a.mini_pacman.i2a_actor_critic import I2A_ActorCritic
from i2a.mini_pacman.i2a_mini_model import I2A_MiniModel
from i2a.mini_pacman.models.i2a_classical_agent import I2A
from i2a.load_utils import load_em_model
from i2a.imagination_core import ImaginationCore

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
        from environment_model.mini_pacman.env_copy_model import CopyEnvModel
        env_model = CopyEnvModel()
    else:
        # the env_model does NOT require grads (require_grad=False) for now, to train jointly set to true
        load_environment_model_dir = 'trained_models/environment_models/'
        if use_class_labels:
            obs_shape = (7, *obs_shape[1:])
            from environment_model.mini_pacman.env_model_label import MiniPacmanEnvModelClassLabels
            EMModel = MiniPacmanEnvModelClassLabels
        else:
            from environment_model.mini_pacman.env_model import MiniPacmanEnvModel
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

