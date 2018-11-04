import torch
import random

class EnvironmentModelTrainer():
    """ Train Environment model
        Args:
            optimizer             : do training update step for environment model
            training_data_creator : function which creates training samples (training_data_creator.create(create_n_samples))
            model_saver           : save model function
            loss_printer          : print loss
                                    loss_printer.log_loss_and_reward(loss=loss,
                                                                     prediction=prediction,
                                                                     sample=batch_sample,
                                                                     episode=i_episode)
        """
    def __init__(self,
                 optimizer,
                 training_data_creator,
                 model_saver,
                 loss_printer,
                 use_cuda = True):
        self.optimizer = optimizer
        self.training_data_creator = training_data_creator
        self.model_saver = model_saver
        self.loss_printer = loss_printer
        self.use_cuda = use_cuda


    def train_episode(self, sample_memory, batch_size, i_episode):
        # sample a state, next-state pair randomly from replay memory for a training step
        batch_sample = [torch.cat(a) for a in
                        zip(*random.sample(sample_memory, batch_size))]
        if self.use_cuda:
            batch_sample = [s.cuda() for s in batch_sample]

        loss, prediction = self.optimizer.optimizer_step(sample=batch_sample)
        # log and print infos
        if self.loss_printer:
            self.loss_printer.log_loss_and_reward(loss=loss,
                                                  prediction=prediction,
                                                  sample=batch_sample,
                                                  episode=i_episode)
        if self.model_saver:
            self.model_saver.save(self.optimizer.model, i_episode)


    def train(self,
              batch_size = 100,
              training_episodes = 10000,
              sample_memory_size = 1000):
        from collections import deque
        print("create training data")
        create_n_samples = min(batch_size * 10, sample_memory_size)
        sample_memory = deque(maxlen = sample_memory_size)
        sample_memory.extend(self.training_data_creator.create(create_n_samples))

        for i_episode in range(training_episodes):
            self.train_episode(sample_memory = sample_memory,
                               batch_size = batch_size,
                               i_episode = i_episode)

            if i_episode != 0 and i_episode % len(sample_memory) == 0:
                print("create more training data")
                sample_memory.extend(self.training_data_creator.create(create_n_samples))

    def train_overfit_on_x_samples(self,
                                   batch_size = 100,
                                   training_episodes = 10000,
                                   x_samples = 100):
        batch_size = min(batch_size, x_samples)
        sample_memory = self.create_x_samples(x_samples)

        for i_episode in range(training_episodes):
            self.train_episode(sample_memory=sample_memory, batch_size=batch_size)
