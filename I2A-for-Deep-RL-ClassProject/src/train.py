from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from ModelAlias import Model
from player_util import Agent
from torch.autograd import Variable
import collections
import cProfile
import time
import datetime


############################


epoch_counter = 0
policy_loss_avg = collections.deque(maxlen= 100)
value_loss_avg = collections.deque(maxlen = 100)

enable_log = True

############################


def train(rank, args, shared_model, optimizer, env_conf):
    #####
    global epoch_counter, policy_loss_avg, value_loss_avg, enable_log
    #######
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    env = atari_env(args.env, env_conf, args)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                [p for p in shared_model.parameters() if p.requires_grad], lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = Model(
        player.env.observation_space.shape[0], player.env.action_space.n, gpu_id >= 0)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
    player.model.train()
    player.eps_len+=2
    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        for step in range(args.num_steps):
            player.action_train()
            if player.done:
                break

        if player.done:
            if player.info['ale.lives'] == 0 or player.max_length:
                player.eps_len = 0
                player.current_life = 0
            state = player.env.reset()
            player.eps_len+=2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _ = player.model(Variable(player.state.unsqueeze(0)))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + args.gamma * \
                player.values[i + 1].data - player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                player.log_probs[i] * \
                Variable(gae) - 0.01 * player.entropies[i]

        ###############################################################
        # custumized loss output
        epoch_counter = (epoch_counter + 1)
        policy_loss_value = policy_loss.data.cpu().numpy()[0][0]
        value_loss_value = value_loss.data.cpu().numpy()[0][0]
        policy_loss_avg.append(policy_loss_value)
        value_loss_avg.append(value_loss_value)

        if (epoch_counter%100) == 0:
            pl_mean = sum(policy_loss_avg) / float(len(policy_loss_avg))
            vl_mean = sum(value_loss_avg) / float(len(value_loss_avg))
            print("Counter: {}, Avg Policy Loss: {}, Avg Value Loss: {}".format(epoch_counter,pl_mean, vl_mean))

        if enable_log == True:
            ts = time.time()
            time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            loss_log_file = open("logs/loss.txt", "a+")
            loss_log_file.write("Train_agent: " + str(rank) + " Time_stemp: " + time_stamp +
                                " Iteration: " + str(epoch_counter) +
                                " policy_loss: " + str(policy_loss_value) +
                                " value_loss: " + str(value_loss_value) + "\n")
            loss_log_file.close()
        ###############################################################

        player.model.zero_grad()
        ''' TODO: do we need to set 'retain_graph=True' in the backward path?'''
        (policy_loss + 0.5 * value_loss).backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm(player.model.parameters(), 40.0)
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()

def profile_train(rank, args, shared_model, optimizer, env_conf):
    cProfile.runctx('train(rank, args, shared_model, optimizer, env_conf)',
                    globals(), locals(), 'prof%d.prof' %rank)