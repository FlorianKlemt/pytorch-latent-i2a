import copy
import glob
import os

import numpy as np
import torch

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from model import Policy
from storage import RolloutStorage, I2A_RolloutStorage
from rl_visualization.visdom_plotter import VisdomPlotterA2C
from a2c_models.a2c_policy_wrapper import A2C_PolicyWrapper

from rl_visualization.play_game_with_trained_model import TestPolicy
import time
import sys
import multiprocessing as mp
import algo
from algo.i2a_algo import I2A_ALGO

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr', 'i2a']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    if args.render_game:
        mp.set_start_method('spawn')

    torch.set_num_threads(1)

    try:
        os.makedirs(args.log_dir)
    except OSError:
        files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
        for f in files:
            if os.path.isfile(f):
                os.remove(f)

    if 'MiniPacman' in args.env_name:
        from environment_model.mini_pacman.builder import MiniPacmanEnvironmentBuilder
        builder = MiniPacmanEnvironmentBuilder(args)
    else:
        from environment_model.latent_space.builder import LatentSpaceEnvironmentBuilder
        builder = LatentSpaceEnvironmentBuilder(args)

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None
        visdom_plotter = VisdomPlotterA2C(viz, args.algo == 'i2a')

    if 'MiniPacman' in args.env_name:
        from gym_envs.envs_mini_pacman import make_custom_env
        envs = [make_custom_env(args.env_name, args.seed, i, args.log_dir, grey_scale=args.grey_scale)
            for i in range(args.num_processes)]
    elif args.algo == 'i2a' or args.train_on_200x160_pixel:
        from gym_envs.envs_ms_pacman import make_env_ms_pacman
        envs = [make_env_ms_pacman(env_id = args.env_name,
                                   seed = args.seed,
                                   rank = i,
                                   log_dir = args.log_dir,
                                   grey_scale = False,
                                   stack_frames = 1,
                                   skip_frames = 4)
                for i in range(args.num_processes)]
    else:
        from envs import make_env
        envs = [make_env(args.env_name, args.seed, i, args.log_dir, args.add_timestep)
                for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    if args.algo == 'i2a' and 'MiniPacman' in args.env_name:
        actor_critic = builder.build_i2a_model(envs, args)
    elif args.algo == 'i2a':
        actor_critic = builder.build_i2a_model(envs, args)
    elif 'MiniPacman' in args.env_name:
        actor_critic = builder.build_a2c_model(envs)
    elif args.train_on_200x160_pixel:
        from a2c_models.atari_model import AtariModel
        actor_critic = A2C_PolicyWrapper(AtariModel(obs_shape=obs_shape,
                                                    action_space=envs.action_space.n,
                                                    use_cuda=args.cuda))
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
        agent = I2A_ALGO(actor_critic=actor_critic, obs_shape=obs_shape, action_shape=action_shape, args=args)
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

    if args.algo == 'i2a':
        rollouts = I2A_RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    else:
        rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space,
                                  actor_critic.state_size)

    current_obs = torch.zeros(args.num_processes, *obs_shape)

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

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            if args.algo  == 'i2a':
                # Sample actions
                value, action, action_log_prob, states, policy_action_prob, rollout_action_prob = actor_critic.act(
                    rollouts.observations[step].clone(),
                    rollouts.states[step],
                    rollouts.masks[step])
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
            if args.algo == "i2a":
                rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks, policy_action_prob, rollout_action_prob)
            else:
                rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.observations[-1],
                                                rollouts.states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        if args.algo == 'i2a':
            value_loss, action_loss, dist_entropy, distill_loss = agent.update(rollouts=rollouts)
        else:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if args.vis:
            distill_loss_data = distill_loss if args.algo == 'i2a' else None
            visdom_plotter.append(dist_entropy,
                                  final_rewards.numpy().flatten(),
                                  value_loss,
                                  action_loss,
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

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps

            reward_info = "mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"\
                .format(final_rewards.mean(), final_rewards.median(), final_rewards.min(), final_rewards.max())

            distill_loss = ", distill_loss {:.5f}".format(distill_loss) if args.algo == 'i2a' else ""
            loss_info = "value loss {:.5f}, policy loss {:.5f}{}"\
                .format(value_loss, action_loss, distill_loss)

            entropy_info = "entropy {:.5f}".format(dist_entropy)

            info = "Updates {}, num timesteps {}, FPS {}, {}, {}, {}, time {:.5f} min"\
                    .format(j, total_num_steps, int(total_num_steps / (end - start)),
                            reward_info, entropy_info, loss_info, (end - start) / 60.)

            with open(log_file, 'a') as the_file:
                the_file.write(info + '\n')

            print(info)
        if args.vis and j % args.vis_interval == 0:
            frames = j*args.num_processes*args.num_steps
            visdom_plotter.plot(frames)




if __name__ == "__main__":
    main()
