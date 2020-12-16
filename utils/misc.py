import os
from itertools import product
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.autograd import Variable
import numpy as np
from progressbar import ProgressBar

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def onehot_from_logits(logits, eps=0.0, dim=1):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(dim, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature, dim=1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=dim)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False, dim=1):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, dim=dim)
    if hard:
        y_hard = onehot_from_logits(y, dim=dim)
        y = (y_hard - y).detach() + y
    return y

def firmmax_sample(logits, temperature, dim=1):
    if temperature == 0:
        return F.softmax(logits, dim=dim)
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)) / temperature
    return F.softmax(y, dim=dim)

def categorical_sample(probs, use_cuda=False):
    int_acs = torch.multinomial(probs, 1)
    if use_cuda:
        tensor_type = torch.cuda.FloatTensor
    else:
        tensor_type = torch.FloatTensor
    acs = Variable(tensor_type(*probs.shape).fill_(0)).scatter_(1, int_acs, 1)
    return int_acs, acs

def disable_gradients(module):
    for p in module.parameters():
        p.requires_grad = False

def enable_gradients(module):
    for p in module.parameters():
        p.requires_grad = True

def sep_clip_grad_norm(parameters, max_norm, norm_type=2):
    """
    Clips gradient norms calculated on a per-parameter basis, rather than over
    the whole list of parameters as in torch.nn.utils.clip_grad_norm.
    Code based on torch.nn.utils.clip_grad_norm
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    for p in parameters:
        if norm_type == float('inf'):
            p_norm = p.grad.data.abs().max()
        else:
            p_norm = p.grad.data.norm(norm_type)
        clip_coef = max_norm / (p_norm + 1e-6)
        if clip_coef < 1:
            p.grad.data.mul_(clip_coef)

def calculate_divergence(dataset, model, n_actions, 
                         mode='random', old_s_divs=None, signs=None):
    n_agents = model.nagents
    if mode != 'recent':
        obs = dataset.obs_buffs # (n_agents, batch_size, o_dim)
    else:
        N = 3000
        if dataset.filled_i == dataset.max_steps:
            inds = np.arange(dataset.curr_i - N, dataset.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, dataset.curr_i - N), dataset.curr_i)
        obs = [dataset.obs_buffs[i][inds] for i in range(n_agents)]
    # acs = np.array(list(product(*([set(tuple(map(tuple, np.eye(n_actions))))] * n_agents)))) # (n_actions ^ n_agents, n_agents, a_dim)
    acs = np.array([np.eye(n_actions).tolist()] * n_agents).transpose(1, 0, 2)
    n_permute = len(acs)
    batch_size = len(obs[0])
    repeated_obs = []
    for a_obs in obs:
        a_obs = Variable(Tensor(a_obs), requires_grad=False)
        repeated_obs.append(a_obs.repeat_interleave(torch.ones(batch_size, dtype=torch.long)*n_permute, dim=0))
    acs = Variable(Tensor(acs), requires_grad=False).permute(1, 0, 2)
    repeated_acs = []
    for a_ac in acs:
        repeated_acs.append(a_ac.repeat(batch_size, 1))

    distances = [0.0] * n_agents
    s_divs = [[] for i in range(n_agents)]
    bar = ProgressBar()    
    for i in bar(range(0, len(repeated_obs[0]), n_permute)):
        if i+n_permute <= len(repeated_obs[0]) - 1:
            sliced_obs = [a_obs[i:i+n_permute].cuda() for a_obs in repeated_obs]
            sliced_acs = [a_ac[i:i+n_permute].cuda() for a_ac in repeated_acs]
        else:
            sliced_obs = [a_obs[i:].cuda() for a_obs in repeated_obs]
            sliced_acs = [a_ac[i:].cuda() for a_ac in repeated_acs]
        critic_in = list(zip(sliced_obs, sliced_acs))
        critic_rets = model.critic(critic_in)
        for a_i, ret in enumerate(critic_rets):
            ret = ret.detach().cpu().numpy()
            # scale to [0, 1]
            # ret = (ret - np.min(ret)) / (np.max(ret) - np.min(ret))
            ret = ret / np.max(ret)
            # TODO: calculate positive value distance and negative value distance
            # remove the maximum value
            # ret = np.expand_dims(ret[ret != np.max(ret)], axis=0)
            # remove zero
            ret = np.expand_dims(ret[ret != 0.0], axis=0)
            min_pos = np.min(ret[ret > 0])
            max_neg = np.max(ret[ret < 0])
            _d = np.sqrt((min_pos - max_neg) ** 2)
            # _d = np.mean(np.sum(np.sqrt((ret - ret.T) ** 2), axis=1))
            distances[a_i] += _d
            s_divs[a_i].append(_d)

    if old_s_divs: 
        if not signs:
            signs = []
        signs.append(np.sign(np.array(s_divs) - np.array(old_s_divs)))

    return [d / batch_size for d in distances], s_divs, signs
