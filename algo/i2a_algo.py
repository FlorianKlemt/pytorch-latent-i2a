import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class I2A_ALGO(object):
    def __init__(self,
                 actor_critic,
                 obs_shape,
                 action_shape,
                 args):

        self.actor_critic = actor_critic
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.args = args

        param = [p for p in actor_critic.parameters() if p.requires_grad]
        self.optimizer = optim.RMSprop(param, args.lr, eps=args.eps, alpha=args.alpha)

    def update(self, rollouts, policy_action_probs, rollout_policy_action_probs):
        values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
                rollouts.observations[:-1].view(-1, *self.obs_shape),
                rollouts.states[0].view(-1, self.actor_critic.state_size),
                rollouts.masks[:-1].view(-1, 1),
                rollouts.actions.view(-1, self.action_shape))

        values = values.view(self.args.num_steps, self.args.num_processes, 1)
        action_log_probs = action_log_probs.view(self.args.num_steps, self.args.num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        # rollout policy optimizer
        rollout_policy_action_log_probs_var = F.log_softmax(rollout_policy_action_probs, dim=2)
        distill_loss = torch.sum(policy_action_probs * rollout_policy_action_log_probs_var, dim=2) #element-wise multiplication in the sum
        distill_loss = distill_loss.mean()

        self.optimizer.zero_grad()
        loss = value_loss * self.args.value_loss_coef + action_loss - dist_entropy * self.args.entropy_coef
        loss = loss + distill_loss * self.args.distill_coef
        loss.backward()

        nn.utils.clip_grad_norm(self.actor_critic.parameters(), self.args.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item(), distill_loss.item()
