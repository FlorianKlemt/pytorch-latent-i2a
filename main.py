import copy
import glob
import os
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym_minipacman

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from model import Policy
from storage import RolloutStorage
from visualize import visdom_plot
from visdom_plotter import VisdomPlotterA2C
from A2C_Models.MiniModel import MiniModel
from A2C_Models.A2C_PolicyWrapper import A2C_PolicyWrapper
from A2C_Models.I2A_MiniModel import I2A_MiniModel
from I2A.I2A_Agent import I2A

from play_game_with_trained_model import TestPolicy

import time
import sys
import multiprocessing as mp
import algo

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr', 'i2a']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    if args.render_game:
        mp.set_start_method('spawn')

    torch.set_num_threads(1)

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None
        visdom_plotter = VisdomPlotterA2C(viz, args.algo == 'i2a')

    if 'MiniPacman' in args.env_name:
        from custom_envs import make_custom_env
        envs = [make_custom_env(args.env_name, args.seed, i, args.log_dir, grey_scale=args.grey_scale)
            for i in range(args.num_processes)]
    else:
        from envs import make_env
        envs = [make_env(args.env_name, args.seed, i, args.log_dir, args.add_timestep)
                for i in range(args.num_processes)]

    #@future self: it might be tempting to move this below after the initialization of envs is finished - dont do it.
    #              SubprovVecEnv hides the unwrapping. At least this incredibly ugly line makes the code 'dynamic' - it's something
    if 'MiniPacman' in args.env_name:
        em_model_reward_bins = envs[0]().unwrapped.reward_bins

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    if args.algo == 'i2a':
        #build i2a model also wraps it with the A2C_PolicyWrapper
        color_prefix = 'grey_scale' if args.grey_scale else 'RGB'
        actor_critic, rollout_policy = build_i2a_model(obs_shape=envs.observation_space.shape,
                                                       frame_stack=args.num_stack,
                                                       action_space=envs.action_space.n,
                                                       em_model_reward_bins=em_model_reward_bins,
                                                       use_cuda=args.cuda,
                                                       environment_model_name=args.env_name + color_prefix + ".dat",
                                                       use_copy_model=args.use_copy_model)

    elif 'MiniPacman' in args.env_name:
        #actor_critic = MiniModel(obs_shape[0], envs.action_space.n, use_cuda=args.cuda)
        actor_critic = A2C_PolicyWrapper(I2A_MiniModel(obs_shape=obs_shape, action_space=envs.action_space.n, use_cuda=args.cuda))
    else:
        actor_critic = Policy(obs_shape, envs.action_space, args.recurrent_policy)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.load_model:
        load_path = os.path.join(args.save_dir, args.algo)
        load_path = os.path.join(load_path, args.env_name + ".pt")
        if os.path.isfile(load_path):
            # if args.cuda:
            saved_state = torch.load(load_path, map_location=lambda storage, loc: storage)
            actor_critic.load_state_dict(saved_state)
        else:
            print("Can not load model ", load_path, ". File does not exists")
            return

    log_file = os.path.join(os.path.join(args.save_dir, args.algo), args.env_name + ".log")
    if not os.path.exists(log_file) or not args.load_model:
        print("Log file: ", log_file)
        with open(log_file, 'w') as the_file:
            the_file.write('command line args: ' + " ".join(sys.argv) + '\n')


    if args.cuda:
        actor_critic.cuda()

    if args.render_game:
        load_path = os.path.join(args.save_dir, args.algo)
        test_process = TestPolicy(model=copy.deepcopy(actor_critic),
                                  load_path=load_path,
                                  args=args)

   if args.algo == 'i2a':
        #param = [p for p in actor_critic.parameters() if (p.requires_grad and not p in rollout_policy.parameters())]
        #nope = [k for k in rollout_policy.parameters()]
        #param = [p for p in param if not p in nope]
        #for p in actor_critic.parameters():
        #    if p.requires_grad:
        #        for a in rollout_policy.parameters():
        #            if p.data.shape==a.data.shape and p.data.eq(a.data):
        #                print("mkay")

        param = [p for p in actor_critic.parameters() if p.requires_grad]
        optimizer = optim.RMSprop(param, args.lr, eps=args.eps, alpha=args.alpha)
    elif args.algo == 'a2c':    
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)
    policy_action_probs = torch.zeros(args.num_steps, args.num_processes, envs.action_space.n)
    # shouldn't it be a Variable??
    rollout_policy_action_probs = []

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()
        policy_action_probs = policy_action_probs.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            if args.algo  == 'i2a':
                # Sample actions
                value, action_prob = actor_critic.policy(Variable(rollouts.observations[step], volatile=True))
                action = actor_critic.sample(action_prob, deterministic=False)
                action_log_prob, _ = actor_critic.logprobs_and_entropy(action_prob, action)
                states = Variable(rollouts.states[step], volatile=True)

                # we need to calculate the destillation loss for the I2A Rollout Policy
                _, rp_actor = rollout_policy(Variable(rollouts.observations[step]))
                rollout_action_prob = F.softmax(rp_actor, dim=1)
                #_, _, rollout_action_prob, _ = rollout_policy.act(Variable(rollouts.observations[step]), None, None)  #NO volatile here, because of distillation loss backprop

                policy_action_probs[step].copy_(action_prob.data)
                #rollout_policy_action_probs[step].copy_(rollout_action_prob.data)
                rollout_policy_action_probs.append(rollout_action_prob)
            else:
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, states = actor_critic.act(
                        rollouts.observations[step],
                        rollouts.states[step],
                        rollouts.masks[step])
            cpu_actions = action.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)
            rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.observations[-1],
                                                rollouts.states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if args.vis:
            distill_loss_data = distill_loss.data[0] if args.algo == 'i2a' else None
            visdom_plotter.append(dist_entropy.data[0],
                                  final_rewards.numpy().flatten(),
                                  value_loss.data[0],
                                  action_loss.data[0],
                                  distill_loss_data)

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            torch.save(save_model.state_dict(), os.path.join(save_path, args.env_name + ".pt"))
            #if args.cuda:
            #    save_model = copy.deepcopy(actor_critic).cpu()
            #save_model = [save_model,
            #                hasattr(envs, 'ob_rms') and envs.ob_rms or None]
            #torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps

            reward_info = "mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"\
                .format(final_rewards.mean(), final_rewards.median(), final_rewards.min(), final_rewards.max())

            distill_loss = ", distill_loss {:.5f}".format(distill_loss.data[0]) if args.algo == 'i2a' else ""
            loss_info = "value loss {:.5f}, policy loss {:.5f}{}"\
                .format(value_loss.data[0], action_loss.data[0], distill_loss)

            entropy_info = "entropy {:.5f}".format(dist_entropy.data[0])

            info = "Updates {}, num timesteps {}, FPS {}, {}, {}, {}, time {:.5f} min"\
                    .format(j, total_num_steps, int(total_num_steps / (end - start)),
                            reward_info, entropy_info, loss_info, (end - start) / 60.)

            with open(log_file, 'a') as the_file:
                the_file.write(info + '\n')

            print(info)
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass
            frames = j*args.num_processes*args.num_steps
            visdom_plotter.plot(frames)



def build_i2a_model(obs_shape, frame_stack, action_space, em_model_reward_bins, use_cuda, environment_model_name, use_copy_model):
    from I2A.EnvironmentModel.MiniPacmanEnvModel import MiniPacmanEnvModel, CopyEnvModel
    from I2A.load_utils import load_em_model
    from I2A.ImaginationCore import ImaginationCore

    input_channels = obs_shape[0]
    obs_shape_frame_stack = (obs_shape[0] * frame_stack, *obs_shape[1:])
    if use_copy_model:
        env_model = CopyEnvModel()
    else:
        # the env_model does NOT require grads (require_grad=False) for now, to train jointly set to true
        load_environment_model_dir = 'trained_models/environment_models/'
        env_model = load_em_model(EMModel=MiniPacmanEnvModel,
                                 load_environment_model_dir=load_environment_model_dir,
                                 environment_model_name=environment_model_name,
                                 obs_shape=obs_shape,
                                 action_space=action_space,
                                 reward_bins=em_model_reward_bins,
                                 use_cuda=use_cuda)

    for param in env_model.parameters():
        param.requires_grad = False
    env_model.eval()

    #TODO: give option to load rollout_policy
    #load_policy_model_dir = os.path.join(os.getcwd(), 'trained_models/a2c/')
    #self.policy = load_policy(load_policy_model_dir=load_policy_model_dir,
    #                          policy_file="RegularMiniPacmanNoFrameskip-v0.pt",
    #                          action_space=action_space,
    #                          use_cuda=use_cuda,
    #                          policy_name="MiniModel")

    #obs_shape = (4, (obs_shape[1:]))   #legacy, if the next line breaks try this

    rollout_policy = A2C_PolicyWrapper(I2A_MiniModel(obs_shape=obs_shape_frame_stack, action_space=action_space, use_cuda=use_cuda))
    for param in rollout_policy.parameters():
        param.requires_grad = True
    rollout_policy.train()

    if use_cuda:
        env_model.cuda()
        rollout_policy.cuda()


    imagination_core = ImaginationCore(env_model=env_model, rollout_policy=rollout_policy,
                                       grey_scale=args.grey_scale, frame_stack=args.num_stack)

    i2a_model = A2C_PolicyWrapper(I2A(obs_shape=obs_shape_frame_stack,
                                      action_space=action_space,
                                      imagination_core=imagination_core,
                                      use_cuda=args.cuda))

    return i2a_model, rollout_policy

if __name__ == "__main__":
    main()
