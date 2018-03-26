from random import shuffle

import numpy as np
import shutil
import torch
from torch.autograd import Variable
import time
import os

DEFAULT_CHECKPOINT_PATH = os.path.join('trained_models', 'model_best.pth.tar')


def print_step_summary_and_update_best_values(epoch, train_loss,
                                              train_accuracy, test_loss, test_accuracy, duration):
    # print table header again after every 3000th step
    if epoch % 100 == 0 and epoch > 0:
        print('Epoch\tTrain Loss\tTrain accuracy\t\tTest Loss\tTest accuracy\tDuration')
    train_string = '{0:.5f}\t\t{1:.2f}%\t\t\t'.format(train_loss, train_accuracy * 100)
    test_string = '{0:.5f}\t\t{1:.2f}%\t\t'.format(test_loss, test_accuracy * 100)

    print('{0}\t'.format(epoch) + train_string + test_string + '{0:.3f}'.format(duration))


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss(), **kwargs):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self.batch_size = kwargs.pop('batch_size', 100)
        self.early_stopping = kwargs.pop('early_stopping', -1)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)
        self.should_use_cuda = kwargs.pop('should_use_cuda', True)

        self.epoch = 0
        self.best_val_acc = 0
        self.best_model_checkpoint = None
        self.early_stopping_counter = self.early_stopping
        self.loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self._reset()

    def _reset(self):
        self.epoch = 0
        self.best_val_acc = 0
        self.best_model_checkpoint = None
        self.early_stopping_counter = self.early_stopping
        self.loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def _step(self, model, input_var, target_var):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        self.optim.zero_grad()

        # Compute loss and gradient
        output = model(input_var)

        # make sure targets are long
        target_var = target_var.long()

        loss = self.loss_func(output, target_var)

        # preform training step
        loss.backward()
        self.optim.step()

        self.loss_history.append(loss.data[0])

    def train(self, model, train_loader, val_loader, num_epochs=10):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        self.optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available() and self.should_use_cuda:
            model.cuda()

        print('START TRAIN.')
        ########################################################################
        #                                                                 #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        num_iterations = num_epochs * iter_per_epoch

        if self.verbose:
            print('Epoch\tTrain Loss\tTrain accuracy\t\tTest Loss\tTest accuracy\tDuration')
            print('=====================================================================================================')
            start_time = time.time()
        continue_training = True
        for epoch in range(num_epochs):
            if not continue_training:
                break

            for t, (input_var, target_var) in enumerate(train_loader, 1):

                if model.is_cuda:
                    input_var, target_var = Variable(input_var.cuda()), Variable(target_var.cuda())
                else:
                    input_var, target_var = Variable(input_var), Variable(target_var)

                self._step(model, input_var, target_var)

                # At the end of every epoch, increment the epoch counter and decay the
                # learning rate.
                epoch_end = (t + 1) % iter_per_epoch == 0
                if epoch_end:
                    self.epoch += 1

                # Check train and val accuracy on the first iteration, the last
                # iteration, and at the end of each epoch.
                first_it = (t == 0)
                last_it = (t == num_iterations + 1)
                if first_it or last_it or epoch_end:
                    train_acc = self.accuracy(model, input_var, target_var)

                    # Compute average valid loss / accuracy
                    avg_val_loss = []
                    avg_val_acc = []
                    for input_val_var, target_val_var in val_loader:

                        if model.is_cuda:
                            input_val_var, target_val_var = Variable(input_val_var.cuda()), Variable(target_val_var.cuda())
                        else:
                            input_val_var, target_val_var = Variable(input_val_var), Variable(target_val_var)

                        avg_val_acc.append(self.accuracy(model, input_val_var, target_val_var))

                        # get valid loss
                        output = model(input_val_var)
                        target_val_var = target_val_var.long()
                        val_loss = self.loss_func(output, target_val_var).data[0]
                        avg_val_loss.append(val_loss)

                    val_loss = sum(avg_val_loss) / len(avg_val_loss)
                    val_acc = sum(avg_val_acc) / len(avg_val_acc)
                    self.val_loss_history.append(val_loss)
                    self.train_acc_history.append(train_acc)
                    self.val_acc_history.append(val_acc)

                    if self.verbose:
                        duration = time.time() - start_time
                        start_time = time.time()

                        # don't take the average in the first iteration
                        if epoch_end:
                            # include all losses from the current epoch
                            start_epoch = t - iter_per_epoch
                            average_train_loss = sum(self.loss_history[start_epoch:]) / iter_per_epoch
                            self.train_loss_history.append(average_train_loss)

                            print_step_summary_and_update_best_values(self.epoch, average_train_loss, train_acc,
                                                                      val_loss, val_acc, duration)
                        else:
                            # then just take the last result
                            self.train_loss_history.append(self.train_loss_history[-1])


                    # early stopping if no improvement of val_acc during the last self.early_stopping epochs
                    # https://link.springer.com/chapter/10.1007/978-3-642-35289-8_5
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc

                        # Save best model
                        self.best_model_checkpoint = {
                            'epoch': self.epoch,
                            'state_dict': model.state_dict(),
                            'val_acc': val_acc,
                            'optimizer': self.optim.state_dict(),
                        }
                        self._save_checkpoint()

                        # restore early stopping counter
                        self.early_stopping_counter = self.early_stopping

                    else:
                        self.early_stopping_counter -= 1

                        # if early_stopping_counter is 0 restore best weights and stop training
                        if self.early_stopping > -1 and self.early_stopping_counter <= 0:
                            print('> Early Stopping after {0} epochs of no improvements.'.format(self.early_stopping))
                            print('> Restoring params of best model with validation accuracy of: '
                                  , self.best_val_acc)

                            # Restore best model
                            model.load_state_dict(self.best_model_checkpoint['state_dict'])
                            self.optim.load_state_dict(self.best_model_checkpoint['optimizer'])
                            continue_training = False
                            break

        print('=====================================================================================================')

        # At the end of training swap the best params into the model
        # Restore best model
        model.load_state_dict(self.best_model_checkpoint['state_dict'])
        self.optim.load_state_dict(self.best_model_checkpoint['optimizer'])

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')

    def accuracy(self, model, inputs, targets, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        targets = targets.long()
        outputs = model.forward(inputs)
        _, preds = torch.max(outputs, 1)
        targets_mask = targets >= 0
        score = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
        return score

        # output = model.forward(sample)
        # _, predicted = torch.max(output, 1)
        # total = model.num_samples(target)
        #
        # # remove parts from the accuracy calculation that are -1 for the segmentation
        # target_mask = target > -1
        # target = target.long()
        #
        # correct = (predicted == target)[target_mask].sum()
        # return correct.data[0] / total

        # maxk = max(topk)
        # batch_size = target.size(0)
        #
        # output = model(sample)
        # _, prediction = output.topk(maxk, 1, True, True)
        # prediction = prediction.t()
        # correct = prediction.eq(target.view(1, -1).expand_as(prediction))
        #
        # res = []
        # for k in topk:
        #     correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        #     res.append(correct_k.mul_(100.0 / batch_size))
        # return res[0]

    def _save_checkpoint(self, filename='checkpoint.pth.tar'):
        torch.save(self.best_model_checkpoint, filename)
        shutil.copyfile(filename, DEFAULT_CHECKPOINT_PATH)
