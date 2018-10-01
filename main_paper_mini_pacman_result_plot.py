import argparse
from rl_visualization.visdom_plotter import VisdomPlotGraph

def extract_values_from_log_line(log_line, algo_i2a = True, reward_multiplier = 1):
    values = log_line.split(', ')
    values = dict([s.rsplit(' ', 1) for s in values])
    update = int(values['Updates'])
    frames = int(values['num timesteps'])
    reward = (values['mean/median reward'].split('/'))
    reward = [float(i) * reward_multiplier for i in reward]
    return frames, reward

def extract_log(log_path):
    with open(log_path) as f:
        log = f.readlines()
    log = [x.strip() for x in log]

    para = log[0].split(' --')
    train_args = [s.rsplit(' ', 1) for s in para[1:]]
    train_args = dict([s for s in train_args if len(s) == 2])

    algo_i2a = train_args['algo'] == 'i2a'
    #reward_multiplier = 1
    #if 'MsPacman' in train_args['env-name']:
    #    reward_multiplier = 10

    log_data = []
    for log_line in log[1:]:
        log_data.append(extract_values_from_log_line(log_line, algo_i2a))

    return log_data

class VisdomPlotterMiniPacmanResults():
    def __init__(self, viz, reward_plot_cnf=[20000, 2000]):
        # from visdom import Visdom
        self.viz = viz
        self.reward_plotter = VisdomPlotGraph(viz,
                                              "Reward",
                                              "Reward",
                                              ["A2C Reward", "Copy Model", "I2A Reward"],
                                              running_mean_n=reward_plot_cnf[0],
                                              plot_after_n_inserts=reward_plot_cnf[1])



    def append(self, a2c_reward, copy_model_reward, i2a_reward):
        self.reward_plotter.extend([[a2c_reward, copy_model_reward, i2a_reward]])

    def plot(self, frames):
        self.reward_plotter.plot(frames)

def main():
    args_parser = argparse.ArgumentParser(description='Make Environment Model arguments')
    args_parser.add_argument('--log-path-a2c', default="trained_models/a2c/HuntMiniPacmanNoFrameskip-v0.log",
                             help='relative path to folder from which a environment model should be loaded.')
    args_parser.add_argument('--log-path-copy-model', default="trained_models/i2a/HuntMiniPacmanNoFrameskip-v0-CopyModel.log",
                             help='relative path to folder from which a environment model should be loaded.')
    args_parser.add_argument('--log-path-i2a', default="trained_models/i2a/HuntMiniPacmanNoFrameskip-v0.log",
                             help='relative path to folder from which a environment model should be loaded.')
    args_parser.add_argument('--port', type=int, default=8097,
                             help='port to run the server on (default: 8097)')
    args_parser.add_argument('--smooth-n-values', type=int, default=3000,
                             help='port to run the server on (default: 8097)')
    args = args_parser.parse_args()



    n = args.smooth_n_values
    from visdom import Visdom
    viz = Visdom(port=args.port)
    visdom_plotter = VisdomPlotterMiniPacmanResults(viz, reward_plot_cnf=[n, 2])

    log_a2c = extract_log(args.log_path_a2c)
    log_copy_model = extract_log(args.log_path_copy_model)
    log_i2a = extract_log(args.log_path_i2a)

    not_finished = True
    i_a2c = 0
    i_copy = 0
    i_i2a = 0
    rewards = []
    infinity = 10000000000
    while(not_finished):
        if i_a2c < len(log_a2c) - 1:
            frame_a2c = log_a2c[i_a2c][0]
            reward_a2c = log_a2c[i_a2c][1][0]
        else:
            frame_a2c = infinity
            reward_a2c = log_a2c[i_a2c-1][1][0]
        if i_copy < len(log_copy_model) - 1:
            frame_copy = log_copy_model[i_copy][0]
            reward_copy = log_copy_model[i_copy][1][0]
        else:
            frame_copy = infinity
            reward_copy = log_copy_model[i_copy-1][1][0]
        if i_i2a < len(log_i2a) - 1:
            frame_i2a = log_i2a[i_i2a][0]
            reward_i2a = log_i2a[i_i2a][1][0]
            if i_i2a > 7813:
                frame_i2a += 99994880
        else:
            frame_i2a = infinity
            reward_i2a = log_i2a[i_i2a-1][1][0]

        if (frame_a2c < frame_copy and frame_a2c < frame_i2a and i_a2c < len(log_a2c) - 1):
            rewards.append([frame_a2c, reward_a2c, reward_copy, reward_i2a])
            i_a2c += 1;
        elif(frame_copy < frame_a2c and frame_copy < frame_i2a and i_copy < len(log_copy_model) - 1):
            rewards.append([frame_copy, reward_a2c, reward_copy, reward_i2a])
            i_copy += 1
        elif(frame_i2a < frame_a2c and frame_i2a < frame_copy and i_i2a < len(log_i2a) - 1):
            rewards.append([frame_i2a, reward_a2c, reward_copy, reward_i2a])
            i_i2a += 1

        if (i_a2c == len(log_a2c) - 1 and i_copy == len(log_copy_model) - 1 and i_i2a == len(log_i2a) - 1):
            not_finished = False

        if (rewards[len(rewards)-1][0] < rewards[len(rewards)-2][0]):
            print("what")

    #54650
    i = 0
    for reward in rewards:
        visdom_plotter.append(reward[1], reward[2], reward[3])
        if (i % 50) == 1:
            visdom_plotter.plot(reward[0])
        i += 1
    print("plot finished")


if __name__ == '__main__':
    main()
