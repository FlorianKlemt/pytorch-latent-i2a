def get_args():
    import argparse
    args_parser = argparse.ArgumentParser(description='Make Environment Model arguments')
    args_parser.add_argument('--load-environment-model', action='store_true', default=False,
                             help='flag to continue training on pretrained env_model')
    args_parser.add_argument('--save-environment-model-dir', default="trained_models/environment_models/",
                             help='relative path to folder from which a environment model should be loaded.')
    args_parser.add_argument('--no-policy-model-loading', action='store_true', default=False,
                             help='use NO pretrained policy model for environment model training')
    args_parser.add_argument('--load-policy-model-dir', default='trained_models/a2c/',
                             help='directory to save agent logs (default: trained_models/a2c)')
    args_parser.add_argument('--env-name', default='RegularMiniPacmanNoFrameskip-v0',
                             help='environment to train on (default: RegularMiniPacmanNoFrameskip-v0)')
    args_parser.add_argument('--environment-model', default='dSSM_DET',
                             help='environment model (default: dSSM_DET)'
                                  'mini pacman models = (MiniModel, MiniModelLabels)'
                                  'latent space models = (dSSM_DET, dSSM_VAE, sSSM)')
    args_parser.add_argument('--skip-frames', type=int, default=4,
                             help='Only used when train with latent space model'
                                  'skip frames (default: 4)')
    args_parser.add_argument('--render', action='store_true', default=False,
                             help='starts an progress that play and render games with the current model')
    args_parser.add_argument('--no-training', action='store_true', default=False,
                             help='true to render a already trained env model, sets load-environment-model and render to True')
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
    args_parser.add_argument('--batch-size', type=int, default=100,
                             help='batch size (default: 100)')
    args_parser.add_argument('--sample-memory-size', type=int, default=50,
                             help='sample memory size (default: 50)')
    args_parser.add_argument('--rollout-steps', type=int, default=10,
                             help='Only used when train with latent space model'
                                  'train with x rollouts (default: 10)')
    args_parser.add_argument('--reward-prediction-bits', type=int, default=8,
                             help='Only used when train with latent space model'
                                  'bits used for reward prediction in reward head of decoder (default: 8)')
    args_parser.add_argument('--num-episodes', type=int, default=10000000,
                             help='number of training episodes (default: 10000000)')
    args_parser.add_argument('--lr', type=float, default=7e-4,
                             help='learning rate (default: 7e-4)')
    args_parser.add_argument('--eps', type=float, default=1e-8,
                             help='RMSprop optimizer epsilon (default: 1e-8)')
    args_parser.add_argument('--weight-decay', type=float, default=0.05,
                             help='weight decay (default: 0.05)')
    args_parser.add_argument('--reward-loss-coef', type=float, default=0.01,
                             help='reward loss coef (default: 0.01)')
    args_parser.add_argument('--env-model-images-save-path', default=None, help='path to save images of rendering')

    args = args_parser.parse_args()

    import torch
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    if args.no_training:
        args.render = True
        args.load_environment_model = True

    return args



def main():
    args = get_args()
    if args.render:
        import multiprocessing as mp
        mp.set_start_method('spawn')

    if 'MiniPacman' in args.env_name:
        from environment_model.mini_pacman.builder import MiniPacmanEnvironmentBuilder
        builder = MiniPacmanEnvironmentBuilder(args)
    else:
        from environment_model.latent_space.builder import LatentSpaceEnvironmentBuilder
        builder = LatentSpaceEnvironmentBuilder(args)


    env = builder.build_env()
    environment_model = builder.build_environment_model(env)
    policy = builder.build_policy(env)

    if args.render:
        test_process = builder.build_environment_model_tester(env, policy, environment_model)

    if args.no_training:
        import time
        time.sleep(100000)
    else:
        trainer = builder.build_environment_model_trainer(env, policy, environment_model)
        trainer.train(batch_size = args.batch_size,
                      training_episodes = args.num_episodes,
                      sample_memory_size = 1000)

    if args.render:
        test_process.stop()



if __name__ == '__main__':
    main()
