from A2C_Models.model import Policy
import torch.nn.functional as F

class A2C_PolicyWrapper(Policy):
    def __init__(self, policy):
        super(A2C_PolicyWrapper, self).__init__()
        self.policy = policy

    def forward(self, inputs, states, masks):
        raise NotImplementedError('This forward should never be called! The policy should only be called via the act, get_value and evaluate functions.'
                                  '@future self: if you read this refactor this whole class')

    def act(self, inputs, states, masks, deterministic=False):
        value, actor = self.policy(inputs)      # value is critic

        action = self.sample(actor, deterministic=deterministic)
        action_log_probs, _ = self.logprobs_and_entropy(actor, action)

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        value, _ = self.policy(inputs)
        return value

    def evaluate_actions(self, inputs, states, masks, actions):
        value, actor = self.policy(inputs)      # value is critic
        action_log_probs, dist_entropy = self.logprobs_and_entropy(actor, actions)

        return value, action_log_probs, dist_entropy, states

    # magic, see model.py
    @property
    def state_size(self):
        return 1


    # in contrast to the method in distributions.py the logprobs_and_entropy and the sample methods have
    # no linear actor layer since this is supposed to be done in the wrapped policy
    def logprobs_and_entropy(self, actor, actions):
        log_probs = F.log_softmax(actor, dim=1)
        probs = F.softmax(actor, dim=1)

        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, dist_entropy

    def sample(self, x, deterministic):
        probs = F.softmax(x, dim=1)
        if deterministic is False:
            action = probs.multinomial(num_samples=1)
        else:
            action = probs.max(1, keepdim=True)[1]
        return action