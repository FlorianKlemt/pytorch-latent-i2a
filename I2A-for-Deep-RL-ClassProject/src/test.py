from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import atari_env
from utils import setup_logger
from ModelAlias import Model
from player_util import Agent
import time
import logging
import cv2
import os
import numpy as np
import sys
import pickle
import datetime


class FrameSaver():
    def __init__(self,
                 save_images_path,
                 save_images_after_xth_iteration,
                 render_window_sizes):
        self.save_images_after_xth_iteration = save_images_after_xth_iteration

        current_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
        self.save_images_path = os.path.join(current_dir, save_images_path)

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', render_window_sizes)

        self.actions = []
        self.rewards = []


    def render_and_save_frame(self, frame,
                              epoch, iteration,
                              save_frame):
        # TODO Find better way for scaling

        frame_data = frame.cpu().numpy()[0]
        frame_data = cv2.normalize(frame_data.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        cv2.imshow('frame', frame_data)
        cv2.waitKey(1)

        if save_frame:
            frame_data *= 255
            frame_data = frame_data.astype(np.uint8)

            image_dir = os.path.join(self.save_images_path, 'game_{0}/'.format(epoch))
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)

            image_path = os.path.join(image_dir,
                                     'game{0}_frame_{1}.png'.format(epoch, iteration))
            cv2.imwrite(image_path, frame_data)

    def log_reward_and_action(self, action, reward, save_frame):
        if save_frame:
            self.actions.append(action)
            self.rewards.append(reward)

    def dump_rewards_and_actions(self, epoch, save_frame):
        if save_frame:
            dump_file = os.path.join(self.save_images_path,
                                         'game_{0}/game{0}_actions_and_rewards.pkl'.format(epoch))
            pickle.dump((self.actions, self.rewards), open(dump_file, "wb"))
        self.actions = []
        self.rewards = []


def test(args, shared_model, env_conf):
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]
    log = {}
    setup_logger('{}_log'.format(args.env),
                 r'{0}{1}_log'.format(args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger(
        '{}_log'.format(args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    env = atari_env(args.env, env_conf, args)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = Model(
        player.env.observation_space.shape[0], player.env.action_space.n, gpu_id >= 0)

    player.state = player.env.reset()
    player.eps_len+=2
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()
    player.model.eval()

    frames = FrameSaver('trained_models/training', 1, render_window_sizes=(400, 400))

    epoch = 0
    save_frame = True

    while True:
        if player.done:
            if gpu_id >= 0:

                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())


        frames.render_and_save_frame(player.state,
                                     epoch,
                                     player.eps_len,
                                     save_frame)

        player.action_test()
        reward_sum += player.reward

        frames.log_reward_and_action(player.action, player.reward, save_frame)

        if player.done:
            frames.dump_rewards_and_actions(epoch, save_frame)

            epoch += 1
            save_frame = epoch % 10 == 0

        if player.done and player.info['ale.lives'] > 0 and not player.max_length:   #ugly hack need to clean this up
            state = player.env.reset()
            player.eps_len+=2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        elif player.done or player.max_length:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_log'.format(args.env)].info(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean))

            ###############################################################
            enable_log = True
            if enable_log == True:
                ts = time.time()
                time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

                reward_log_file = open("logs/reward.txt", "a+")
                reward_log_file.write("time_stemp: " + time_stamp + " episode_lenght: " + str(player.eps_len) +
                                      " reward: " + str(reward_sum) + "\n")
                reward_log_file.close()
            ###############################################################

            if reward_sum > args.save_score_level:
                score_level = int(reward_sum)
                player.model.load_state_dict(shared_model.state_dict())
                state_to_save = player.model.state_dict()
                torch.save(state_to_save, '{0}{1}_{2}.dat'.format(
                    args.save_model_dir, args.env, score_level))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            player.eps_len+=2
            time.sleep(10)

            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

