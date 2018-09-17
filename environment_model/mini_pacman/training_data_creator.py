import torch
import torch.nn.functional as F
import random

class TrainingDataCreator:
    def __init__(self, env, policy, use_cuda):
        self.env = env
        self.policy = policy
        self.use_cuda = use_cuda
        self.state = self._env_reset()
        self.done = False

    def _env_reset(self):
        state = self.env.reset()
        state = torch.from_numpy(state).unsqueeze(0).float()
        if self.use_cuda:
            state = state.cuda()
        return state

    def _do_env_step(self, action):
        next_state, reward, done, info = self.env.step(action.item())
        next_state = torch.from_numpy(next_state).unsqueeze(0).float()
        reward = torch.FloatTensor([reward])
        if self.use_cuda:
            next_state = next_state.cuda()
            reward = reward.cuda()
        return  next_state, reward, done, info

    def _sample_action(self, state):
        if self.policy:
            # let policy decide on next action and perform it
            critic, actor = self.policy(state)
            prob = F.softmax(actor, dim=1)
            action = prob.multinomial(num_samples=1)
        else:
            action_space = self.env.action_space.n
            action_int = random.randint(0, action_space - 1)
            action = torch.LongTensor([action_int]).unsqueeze(0)
            if self.use_cuda:
                action = action.cuda()
        return action


    def create(self, number_of_samples):
        from collections import deque
        sample_memory = deque(maxlen=number_of_samples)

        while len(sample_memory) < number_of_samples:
            if self.done:
                self.state = self._env_reset()
                self.done = False

            while not self.done:
                action = self._sample_action(state = self.state)
                next_state, reward, self.done, _ = self._do_env_step(action=action)

                # add current state, next-state pair to replay memory
                sample_memory.append([self.state, action, next_state, reward])
                self.state = next_state

                if len(sample_memory) >= number_of_samples:
                    break

        return sample_memory