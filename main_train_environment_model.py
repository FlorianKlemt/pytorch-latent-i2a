def get_args():
    import argparse
    args_parser = argparse.ArgumentParser(description='Make Environment Model arguments')
    args_parser.add_argument('--load-environment-model', action='store_true', default=False,
                             help='flag to continue training on pretrained env_model')
    args_parser.add_argument('--save-environment-model-dir', default="trained_models/environment_models/",
                             help='relative path to folder from which a environment model should be loaded.')
    args_parser.add_argument('--no-policy-model-loading', action='store_true', default=False,
                             help='use trained policy model for environment model training')
    args_parser.add_argument('--load-policy-model-dir', default='trained_models/a2c/',
                             help='directory to save agent logs (default: trained_models/a2c)')
    args_parser.add_argument('--load-policy-model-name', default='RegularMiniPacmanNoFrameskip-v0',
                             help='directory to save agent logs (default: RegularMiniPacmanNoFrameskip-v0)')
    args_parser.add_argument('--env-name', default='RegularMiniPacmanNoFrameskip-v0',
                             help='environment to train on (default: RegularMiniPacmanNoFrameskip-v0)')
    args_parser.add_argument('--render', action='store_true', default=False,
                             help='starts an progress that play and render games with the current model')
    args_parser.add_argument('--no-training', action='store_true', default=False,
                             help='true to render a already trained env model')
    args_parser.add_argument('--no-cuda', action='store_true', default=False,
                             help='disables CUDA training')
    args_parser.add_argument('--no-vis', action='store_true', default=False,
                             help='disables visdom visualization')
    args_parser.add_argument('--port', type=int, default=8097,
                             help='port to run the server on (default: 8097)')
    args_parser.add_argument('--grey_scale', action='store_true', default=False,
                             help='True to convert to grey_scale images')
    args_parser.add_argument('--save-interval', type=int, default=100,
                             help='save model each n episodes (default: 10)')
    args_parser.add_argument('--use-class-labels', action='store_true', default=False,
                             help='true to use pixelwise cross-entropy-loss and make'
                                  'the color of each pixel a classification task')
    args_parser.add_argument('--batch-size', type=int, default=100,
                             help='batch size (default: 100)')
    args_parser.add_argument('--training-episodes', type=int, default=1000000,
                             help='number of training episodes (default: 1000000)')
    args_parser.add_argument('--lr', type=float, default=7e-4,
                             help='learning rate (default: 7e-4)')
    args_parser.add_argument('--eps', type=float, default=1e-8,
                             help='RMSprop optimizer epsilon (default: 1e-8)')
    args_parser.add_argument('--weight-decay', type=float, default=0.05,
                             help='weight decay (default: 0)')
    args_parser.add_argument('--reward-loss-coef', type=float, default=0.01,
                             help='reward loss coef (default: 0.01)')
    args_parser.add_argument('--env-model-images-save-path', default=None, help='path to save images of rendering')

    args = args_parser.parse_args()

    import torch
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    if args.no_training:
        args.render = True

    return args



def main():
    args = get_args()
    if args.render:
        import multiprocessing as mp
        mp.set_start_method('spawn')

    save_model_path = get_save_model_path(args)


    from gym_envs.envs_mini_pacman import make_custom_env
    env = make_custom_env(args.env_name, seed=1, rank=1, log_dir=None, grey_scale=args.grey_scale)() #wtf


    environment_model = build_environment_model(env = env,
                                                args = args,
                                                load_environment_model_path = save_model_path)

    optimizer = build_optimizer(environment_model = environment_model, args = args)

    policy = build_policy(env=env, args=args)

    if args.render:
        from rl_visualization.environment_model.test_environment_model import TestEnvironmentModelMiniPacman
        import copy
        test_process = TestEnvironmentModelMiniPacman(env = env,
                                                      environment_model=copy.deepcopy(environment_model),
                                                      load_path=save_model_path,
                                                      rollout_policy=policy,
                                                      args=args)

    if args.no_training:
        import time
        time.sleep(100000)
    else:
        # Training Data Creator
        from environment_model.training_data_creator import TrainingDataCreator
        data_creator = TrainingDataCreator(env = env,
                                           policy = policy,
                                           use_cuda = args.cuda)
        # Model Saver
        from environment_model.model_saver import ModelSaver
        model_saver = ModelSaver(save_model_path = save_model_path,
                                 save_interval = args.save_interval)

        # Loss Printer
        loss_printer = build_loss_printer(args = args, batch_size = args.batch_size)

        from environment_model.env_model_trainer import EnvironmentModelTrainer
        trainer = EnvironmentModelTrainer(optimizer=optimizer,
                                          training_data_creator = data_creator,
                                          model_saver = model_saver,
                                          loss_printer=loss_printer,
                                          use_cuda=args.cuda)

        trainer.train(batch_size = args.batch_size,
                      training_episodes = args.training_episodes,
                      sample_memory_size = 1000)
        #trainer.train_overfit_on_x_samples(1000000, x_samples=100)

    if args.render:
        test_process.stop()




def get_save_model_path(args):
    from environment_model.model_saver import save_environment_model_path
    return save_environment_model_path(args.save_environment_model_dir,
                                       args.env_name,
                                       args.use_class_labels,
                                       args.grey_scale)
def get_log_path(args):
    from environment_model.model_saver import save_environment_model_log_path
    return save_environment_model_log_path(args.save_environment_model_dir,
                                           args.env_name,
                                           args.use_class_labels,
                                           args.grey_scale)

def build_loss_printer(args, batch_size):
    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
    else:
        viz = None

    log_path = get_log_path(args)

    from environment_model.visualizer.env_mini_pacman_logger import LoggingMiniPacmanEnvTraining
    loss_printer = LoggingMiniPacmanEnvTraining(log_name = log_path,
                                                batch_size = batch_size,
                                                delete_log_file = args.load_environment_model == False,
                                                viz=viz)
    return loss_printer

def build_environment_model(env,
                            args,
                            load_environment_model_path=None):
    if args.use_class_labels:
        from environment_model.mini_pacman.env_model_label import MiniPacmanEnvModelClassLabels
        EMModel = MiniPacmanEnvModelClassLabels
        labels = 7
        em_obs_shape = (labels, env.observation_space.shape[1], env.observation_space.shape[2])
    else:
        from environment_model.mini_pacman.env_model import MiniPacmanEnvModel
        EMModel = MiniPacmanEnvModel
        em_obs_shape = env.observation_space.shape

    reward_bins = env.unwrapped.reward_bins #[0., 1., 2., 5., 0.] for regular

    environment_model = EMModel(obs_shape=em_obs_shape, #env.observation_space.shape,  # 4
                                num_actions=env.action_space.n,
                                reward_bins=reward_bins,
                                use_cuda=args.cuda)

    if args.load_environment_model:
        import torch
        print("Load environment model", load_environment_model_path)
        saved_state = torch.load(load_environment_model_path, map_location=lambda storage, loc: storage)
        environment_model.load_state_dict(saved_state)
    else:
        print("Save environment model under", load_environment_model_path)

    if args.cuda:
        environment_model.cuda()

    return environment_model

def build_optimizer(environment_model, args):
    if args.use_class_labels:
        from environment_model.mini_pacman.env_optimizer_label import EnvMiniPacmanLabelsOptimizer
        optimizer_type = EnvMiniPacmanLabelsOptimizer
    else:
        from environment_model.mini_pacman.env_optimizer import EnvMiniPacmanOptimizer
        optimizer_type = EnvMiniPacmanOptimizer

    optimizer = optimizer_type(model=environment_model,
                               reward_loss_coef = args.reward_loss_coef,
                               lr = args.lr,
                               eps = args.eps,
                               weight_decay = args.weight_decay,
                               use_cuda = args.cuda)
    return optimizer


def build_policy(env, args):
    load_policy_model_path = '{0}{1}.pt'.format(args.load_policy_model_dir, args.load_policy_model_name)

    from i2a.mini_pacman.i2a_mini_model import I2A_MiniModel
    from a2c_models.a2c_policy_wrapper import A2C_PolicyWrapper
    policy = A2C_PolicyWrapper(I2A_MiniModel(obs_shape=env.observation_space.shape,
                                             action_space=env.action_space.n,
                                             use_cuda=args.cuda))
    if args.cuda:
        policy.cuda()

    if not args.no_policy_model_loading:
        import torch
        saved_state = torch.load(load_policy_model_path, map_location=lambda storage, loc: storage)
        print("Load Policy Model", load_policy_model_path)
        policy.load_state_dict(saved_state)

    return policy


if __name__ == '__main__':
    main()
