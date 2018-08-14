import torch
import torch.nn.functional as F
import random
import numpy as np

class TrainingDataCreator:
    def __init__(self,
                 env,
                 policy,
                 rollouts,
                 initial_context_size,
                 frame_stack_size,
                 sample_memory_on_gpu,
                 use_cuda):
        self.env = env
        self.policy = policy
        self.rollouts = rollouts
        self.initial_context_size = initial_context_size
        self.frame_stack_size = frame_stack_size
        self.sample_memory_on_gpu = sample_memory_on_gpu
        self.use_cuda = use_cuda
        self.frame_stack = self._env_reset()
        self.done = False

    def _env_reset(self):
        state = self.env.reset()
        state = torch.from_numpy(state).unsqueeze(0).float()

        frame_stack = state.repeat(1, self.frame_stack_size, 1, 1)
        if self.sample_memory_on_gpu:
            frame_stack = frame_stack.cuda()
        return frame_stack

    def _stack_frame(self, state):
        # shape frame stack: (1, rgb * stack, w, h)
        # 3 channels for rgb -> remove oldes frame (3 channels) and add state
        self.frame_stack = torch.cat((self.frame_stack[:, 3:], state), dim=1)

    def _initial_steps(self, min, max):
        from random import randint
        for i in range(randint(1, 100)):
            stack = self.frame_stack
            action = self._sample_action(frame_stack=stack)
            state, reward, done, _ = self._do_env_step(action=action)
            self._stack_frame(state)

    def _do_env_step(self, action):
        next_state, reward, done, info = self.env.step(action.item())
        next_state = torch.from_numpy(next_state).unsqueeze(0).float()
        reward = torch.FloatTensor([reward])
        if self.sample_memory_on_gpu:
            next_state = next_state.cuda()
            reward = reward.cuda()
        return  next_state, reward, done, info

    def _sample_action(self, frame_stack):
        if self.policy:
            # let policy decide on next action and perform it
            if self.use_cuda:
                frame_stack = frame_stack.cuda()
            value, action, _, _ = self.policy.act(inputs=frame_stack, states=None, masks=None)  # no state and mask
        else:
            action_space = self.env.action_space.n
            action_int = random.randint(0, action_space - 1)
            action = torch.from_numpy(np.array([action_int])).unsqueeze(0)
            if self.sample_memory_on_gpu:
                action = action.cuda()
        return action


    def create(self, number_of_samples):
        from collections import deque
        sample_memory = deque(maxlen=number_of_samples)

        while len(sample_memory) < number_of_samples:
            if self.done:
                self.frame_stack = self._env_reset()
                self.done = False
                self._initial_steps(0, 200)

            while not self.done:
                initial_context_stack = []
                action_stack = []
                reward_stack = []
                target_state_stack = []

                for i in range(self.initial_context_size):
                    action = self._sample_action(frame_stack=self.frame_stack)
                    state, reward, self.done, _ = self._do_env_step(action=action)
                    initial_context_stack.append(state)
                    self._stack_frame(state)


                for i in range(self.rollouts):
                    action = self._sample_action(frame_stack=self.frame_stack)
                    state, reward, self.done, _ = self._do_env_step(action=action)
                    target_state_stack.append(state)
                    action_stack.append(action)
                    reward_stack.append(reward)
                    self._stack_frame(state)

                #unsqueeze initial_context, action, target_state and reward stack for batch dimension
                sample_memory.append([torch.stack(initial_context_stack, 1),
                                      torch.stack(action_stack, 1),
                                      torch.stack(target_state_stack, 1),
                                      torch.stack(reward_stack, 1)])


                if len(sample_memory) >= number_of_samples:
                    break
        return sample_memory