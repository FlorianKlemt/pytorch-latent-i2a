


class LatentSpaceEnvironmentBuilder():
    def __init__(self, args):
        self.args = args
        self.save_model_path = self.get_save_model_path()
        self.environment_model_name = args.env_name + "_" + args.environment_model + ".dat"
        self.load_environment_model_dir = 'trained_models/environment_models/'
        self.load_environment_model_path = self.load_environment_model_dir + self.environment_model_name

    def get_save_name(self):
        return '{0}_{1}'.format(self.args.env_name,
                                self.args.environment_model)

    def get_save_model_path(self):
        return '{0}{1}.dat'.format(self.args.save_environment_model_dir,
                                   self.get_save_name())
    def get_log_path(self):
        return '{0}{1}.log'.format(self.args.save_environment_model_dir,
                                   self.get_save_name())

    def build_env(self):
        from gym_envs.envs_ms_pacman import make_env_ms_pacman
        env = make_env_ms_pacman(env_id=self.args.env_name,
                                 seed=1, rank=1,
                                 log_dir=None,
                                 grey_scale=False,
                                 skip_frames=self.args.skip_frames,
                                 stack_frames=1)()
        return env

    def build_environment_model(self, env):
        from environment_model.latent_space.models_from_paper.state_space_model import SSM
        environment_model = SSM(model_type=self.args.environment_model,
                                obs_shape=env.observation_space.shape,
                                state_input_channels=64,
                                num_actions=env.action_space.n,
                                use_cuda=self.args.cuda)#,
                                #reward_prediction_bits=self.args.reward_prediction_bits)

        if self.args.load_environment_model:
            import torch
            print("Load environment model", self.save_model_path)
            saved_state = torch.load(self.load_environment_model_path,
                                     map_location=lambda storage, loc: storage)
            environment_model.load_state_dict(saved_state)
        else:
            print("Save environment model under", self.save_model_path)

        if self.args.cuda:
            environment_model.cuda()

        return environment_model

    def build_policy(self, env):
        from a2c_models.a2c_policy_wrapper import A2C_PolicyWrapper
        from a2c_models.atari_model import AtariModel

        obs_shape = (env.observation_space.shape[0] * 4,) + env.observation_space.shape[1:]
        policy = A2C_PolicyWrapper(AtariModel(obs_shape=obs_shape,
                                              action_space=env.action_space.n,
                                              use_cuda=self.args.cuda))

        if self.args.cuda:
            policy.cuda()

        if not self.args.no_policy_model_loading:
            import torch
            load_policy_model_path = '{0}{1}.pt'.format(self.args.load_policy_model_dir, self.args.env_name)
            saved_state = torch.load(load_policy_model_path, map_location=lambda storage, loc: storage)
            print("Load Policy Model", load_policy_model_path)
            policy.load_state_dict(saved_state)
        return policy

    def build_loss_printer(self):
        if self.args.vis:
            from visdom import Visdom
            viz = Visdom(port=self.args.port)
        else:
            viz = None

        log_path = self.get_log_path()

        #from rl_visualization.logger import LogTrainEM
        #loss_printer = LogTrainEM(log_name=log_path,
        #                          batch_size=self.args.batch_size,
        #                          delete_log_file=self.args.load_environment_model == False,
        #                          viz=viz)
        from environment_model.visualizer.env_mini_pacman_logger import LoggingMiniPacmanEnvTraining
        loss_printer = LoggingMiniPacmanEnvTraining(log_name=log_path,
                                                    batch_size=self.args.batch_size,
                                                    delete_log_file=self.args.load_environment_model == False,
                                                    viz=viz)
        return loss_printer

    def build_optimizer(self, environment_model):
        if self.args.environment_model == "dSSM_DET":
            from environment_model.latent_space.dSSM_DET_optimizer import dSSM_DET_Optimizer
            optimizer_type = dSSM_DET_Optimizer
        else:
            from environment_model.latent_space.state_space_optimizer import EnvLatentSpaceOptimizer
            optimizer_type = EnvLatentSpaceOptimizer

        optimizer = optimizer_type(model=environment_model,
                                   reward_prediction_bits=self.args.reward_prediction_bits,
                                   reward_loss_coef=self.args.reward_loss_coef,
                                   lr=self.args.lr,
                                   eps=self.args.eps,
                                   weight_decay=self.args.weight_decay,
                                   use_cuda=self.args.cuda)
        return optimizer

    def build_environment_model_trainer(self, env, policy, environment_model):
        # Training Data Creator
        from environment_model.latent_space.latent_space_training_data_creator import TrainingDataCreator
        data_creator = TrainingDataCreator(env=env,
                                           policy=policy,
                                           rollouts = self.args.rollout_steps,
                                           initial_context_size = 3,
                                           frame_stack_size = 4,
                                           sample_memory_on_gpu = False,
                                           use_cuda=self.args.cuda)
        # Model Saver
        from environment_model.model_saver import ModelSaver
        model_saver = ModelSaver(save_model_path=self.save_model_path,
                                 save_interval=self.args.save_interval)

        # Loss Printer
        loss_printer = self.build_loss_printer()

        optimizer = self.build_optimizer(environment_model=environment_model)

        from environment_model.env_model_trainer import EnvironmentModelTrainer
        trainer = EnvironmentModelTrainer(optimizer=optimizer,
                                          training_data_creator=data_creator,
                                          model_saver=model_saver,
                                          loss_printer=loss_printer,
                                          use_cuda=self.args.cuda)
        return trainer

    def build_environment_model_tester(self, env, policy, environment_model):
        from rl_visualization.environment_model.test_environment_model import TestEnvironmentModel
        import copy
        self.args.use_latent_space = True
        test_process = TestEnvironmentModel(env=copy.deepcopy(env),
                                            environment_model=copy.deepcopy(environment_model),
                                            load_path=self.save_model_path,
                                            rollout_policy=policy,
                                            args=self.args)
        return test_process


    def build_rollout_policy(self, action_space, use_cuda):
        from i2a.rollout_policy import RolloutPolicy
        rollout_policy = RolloutPolicy(obs_shape=(64, 25, 20), action_space=action_space)

        if use_cuda:
            rollout_policy.cuda()
        return rollout_policy

    def build_a2c_model(self, env):
        from i2a.mini_pacman.i2a_mini_model import I2A_MiniModel
        from a2c_models.a2c_policy_wrapper import A2C_PolicyWrapper
        return A2C_PolicyWrapper(
            I2A_MiniModel(obs_shape=env.observation_space.shape,
                          action_space=env.action_space.n,
                          use_cuda=self.args.cuda))

    def build_i2a_model(self,
                        env,
                        args):
        action_space = env.action_space.n
        obs_shape = (env.observation_space.shape[0] * args.num_stack, *env.observation_space.shape[1:])

        environment_model = self.build_environment_model(env)
        for param in environment_model.parameters():
            param.requires_grad = False
        environment_model.eval()

        rollout_policy = self.build_rollout_policy(action_space, args.cuda)
        for param in rollout_policy.parameters():
            param.requires_grad = True
        rollout_policy.train()

        from i2a.latent_space.imagination_core.latent_space_imagination_core import LatentSpaceImaginationCore
        imagination_core = LatentSpaceImaginationCore(env_model=environment_model,
                                                      rollout_policy=rollout_policy,
                                                      grey_scale=args.grey_scale, frame_stack=args.num_stack)

        from i2a.latent_space.models.i2a_agent import I2ALatentSpace
        from i2a.latent_space.i2a_latent_space_actor_critic import I2ALatentSpaceActorCritic

        i2a_model = I2ALatentSpaceActorCritic(policy=I2ALatentSpace(obs_shape=obs_shape,
                                                                    action_space=action_space,
                                                                    imagination_core=imagination_core,
                                                                    rollout_steps=args.i2a_rollout_steps,
                                                                    use_cuda=args.cuda),
                                              imagination_core=imagination_core,
                                              frame_stack=args.num_stack)

        return i2a_model
