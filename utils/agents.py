import torch as th
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.policies import DiscretePolicy

class AttentionAgent(object):
    """
    General class for Attention agents (policy, target policy)
    """
    def __init__(self, num_in_pol, num_out_pol, hidden_dim=64,
                 lr=0.01, onehot_dim=0):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
        """
        self.policy = DiscretePolicy(num_in_pol, num_out_pol,
                                     hidden_dim=hidden_dim)
        self.target_policy = DiscretePolicy(num_in_pol,
                                            num_out_pol,
                                            hidden_dim=hidden_dim)
        self.old_policy = DiscretePolicy(num_in_pol,
                                            num_out_pol,
                                            hidden_dim=hidden_dim)
        self.modeling_policy = DiscretePolicy(num_in_pol,
                                            num_out_pol,
                                            hidden_dim=hidden_dim,
                                            onehot_dim=onehot_dim)
        self.local_trust_region = th.tensor(0.01, requires_grad=True)
        hard_update(self.target_policy, self.policy)
        hard_update(self.old_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.modeling_policy_optimizer = Adam(
            self.modeling_policy.parameters(), lr=lr)

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        return self.policy(obs, sample=explore)

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'old_policy': self.old_policy.state_dict(),
                'modeling_policy': self.modeling_policy.state_dict(),
                'local_trust_region': self.local_trust_region,
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'modeling_policy_optimizer': \
                    self.modeling_policy_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.old_policy.load_state_dict(params['old_policy'])
        self.modeling_policy.load_state_dict(params['modeling_policy'])
        self.local_trust_region = th.load(params['local_trust_region'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.modeling_policy_optimizer.load_state_dict(
            params['modeling_policy_optimizer'])
