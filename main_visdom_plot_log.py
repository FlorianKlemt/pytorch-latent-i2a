import argparse
from rl_visualization.visdom_plotter import VisdomPlotterA2C


def main():
    args_parser = argparse.ArgumentParser(description='Make Environment Model arguments')
    args_parser.add_argument('--log-path', default="trained_models/i2a/HuntMiniPacmanNoFrameskip-v0.log",
                             help='relative path to folder from which a environment model should be loaded.')
    args_parser.add_argument('--port', type=int, default=8097,
                             help='port to run the server on (default: 8097)')
    args_parser.add_argument('--smooth-n-values', type=int, default=3000,
                             help='port to run the server on (default: 8097)')
    args = args_parser.parse_args()

    log_path = args.log_path
    with open(log_path) as f:
        log = f.readlines()
    log = [x.strip() for x in log]

    para = log[0].split(' --')
    train_args = [s.rsplit(' ', 1) for s in para[1:]]
    train_args = dict([s for s in train_args if len(s) == 2])

    algo_i2a =  train_args['algo'] == 'i2a'
    n = args.smooth_n_values
    from visdom import Visdom
    viz = Visdom(port=args.port)
    visdom_plotter = VisdomPlotterA2C(viz, plot_distill_loss = algo_i2a,
                 entropy_plot_cnf = [n, 0], reward_plot_cnf=[n, 0], loss_plot_cnf = [n, 0])

    i = 0
    for log_line in log[1:]:
        values = log_line.split(', ')
        values = dict([s.rsplit(' ', 1) for s in values])
        update = int(values['Updates'])
        frames = int(values['num timesteps'])
        dist_entropy = float(values['entropy'])
        reward = (values['mean/median reward'].split('/'))
        reward = [float(i)*10 for i in reward]
        value_loss = float(values['value loss'])
        action_loss = float(values['policy loss'])
        distill_loss = float(values['distill_loss']) if algo_i2a else None
        visdom_plotter.append(dist_entropy, reward, value_loss, action_loss, distill_loss)
        if (i % 50) == 0:
            visdom_plotter.plot(frames)
        i += 1
    print("plot finished")


if __name__ == '__main__':
    main()
