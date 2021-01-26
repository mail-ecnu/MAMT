import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients, th_log_q
from utils.agents import AttentionAgent
from utils.critics import AttentionCritic
from utils.networks import TRAN
import numpy as np

MSELoss = torch.nn.MSELoss()
HuberLoss = torch.nn.SmoothL1Loss()

class AttentionSACTra(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 tr_scale=10.,
                 tsallis_q=0.5,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4,
                 **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        self.nagents = len(sa_size)

        self.agents = [AttentionAgent(lr=pi_lr,
                                      hidden_dim=pol_hidden_dim,
                                      onehot_dim=self.nagents,
                                      **params)
                         for params in agent_init_params]
        self.critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads)
        self.target_critic = AttentionCritic(
            sa_size, hidden_dim=critic_hidden_dim, attend_heads=attend_heads)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=1e-3)
        self.local_tr_optimizer = Adam(self.local_trs, lr=q_lr, weight_decay=1e-3)
        self.tran = TRAN(sa_size, hidden_dim=critic_hidden_dim)
        self.tran_optimizer = Adam(self.tran.parameters(), lr=q_lr, weight_decay=1e-3)
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.tr_scale = tr_scale
        self.tsallis_q = tsallis_q

        self.ccs = None

        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.old_pol_dev = 'cpu'  # device for old policies
        self.model_pol_dev = 'cpu'
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.tran_dev = 'cpu'
        self.niter = 0


    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    @property
    def old_policies(self):
        return [a.old_policy for a in self.agents]

    @property
    def modeling_policies(self):
        return [a.modeling_policy for a in self.agents]

    @property
    def local_trs(self):
        return [a.local_trust_region for a in self.agents]

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(
            self.agents, observations)]

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def update_all_olds(self):
        """
        Update all old policy networks (called before normal policy updates have been
        performed for each agent)
        """
        for a in self.agents:
            hard_update(a.old_policy, a.policy)

    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        obs, acs, rews, next_obs, dones = sample
        # Q loss
        next_acs = []
        next_log_pis = []
        for pi, ob in zip(self.target_policies, next_obs):
            curr_next_ac, curr_next_log_pi = pi(ob, return_log_pi=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in, regularize=True,
                                  logger=logger, niter=self.niter)
        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.nagents), next_qs,
                                               next_log_pis, critic_rets):
            target_q = (rews[a_i].view(-1, 1) +
                        self.gamma * nq *
                        (1 - dones[a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg  # regularizing attention
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm(
            self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_policies(self, sample, soft=True, tr=False, logger=None, **kwargs):
        obs, acs, rews, next_obs, dones = sample
        samp_acs = []
        all_probs = []
        all_log_pis = []
        all_old_log_pis = []
        all_pol_regs = []

        for a_i, pi, old_pi, ob in zip(
            range(self.nagents), self.policies, self.old_policies, obs):
            curr_ac, probs, log_pi, pol_regs, ent = pi(
                ob, return_all_probs=True, return_log_pi=True,
                regularize=True, return_entropy=True)
            logger.add_scalar('agent%i/policy_entropy' % a_i, ent,
                              self.niter)

            old_log_pi = old_pi.logp_ac(ob, curr_ac)
            kl_delta = (log_pi - old_log_pi).mean()
            logger.add_scalar('agent%i/kl_delta' % a_i, kl_delta,
                              self.niter)

            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_old_log_pis.append(old_log_pi)
            all_pol_regs.append(pol_regs)

        if (self.niter + 1) % 100 == 0:
            self.update_all_olds()
        
        pol_losses = []
        counter_qs = []
        qs = []

        critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in, return_all_q=True)
        for a_i, probs, log_pi, old_log_pi, pol_regs, (q, all_q) in zip(
            range(self.nagents), all_probs, all_log_pis, all_old_log_pis,
            all_pol_regs, critic_rets):
            curr_agent = self.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v

            counter_qs.append(v.clone().detach())
            qs.append(q.clone().detach())

            if soft:
                if tr:
                    q_log_pi = th_log_q(torch.exp(log_pi), q=self.tsallis_q)
                    q_old_log_pi = th_log_q(torch.exp(old_log_pi), q=self.tsallis_q)
                    pol_loss = (log_pi * ((q_log_pi - q_old_log_pi) / self.local_trs[a_i] - pol_target).detach()).mean()
                else:
                    pol_loss = (log_pi * ((log_pi) / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # policy regularization
            pol_losses.append(pol_loss)

        # Calculate coordination coefficients
        self.ccs = torch.abs(torch.stack(qs).permute(1, 0, 2) - \
                             torch.stack(counter_qs).permute(1, 2, 0))
        masks = torch.arange(0, self.nagents, out=torch.LongTensor())
        self.ccs[:, masks, masks] = -np.inf
        self.ccs = F.softmax(self.ccs, dim=1)

        # Trust region assignment
        tran_in = list(zip(obs, [samp_ac.detach() for samp_ac in samp_acs]))
        kl_hat = self.tran(tran_in, self.local_trs, self.ccs)

        # find the agent with largest coordination coefficient
        # to calculate the non-stationarity
        tightest_as = torch.max(torch.sum(
            self.ccs, dim=0), dim=1)[1].detach().cpu().numpy()
        # calculate the non-stationarity
        batch_size = len(obs[0])
        d_nses = []
        for a_i in range(self.nagents):
            opponent_id = torch.LongTensor(batch_size, 1).fill_(tightest_as[a_i])
            opponent_onehot = torch.FloatTensor(
                batch_size, self.nagents).zero_().scatter_(1, opponent_id, 1).cuda()
            _, log_pi = self.modeling_policies[a_i](
                (obs[a_i], opponent_onehot), return_all_log_probs=True)
            _, pi = self.policies[tightest_as[a_i]](
                obs[tightest_as[a_i]], return_all_probs=True)
            d_ns = F.kl_div(log_pi, pi, reduction='none').mean(dim=1).unsqueeze(-1)
            d_nses.append(d_ns)
        d_nses = torch.cat(d_nses, dim=1)
        # Update trust region assignment network
        # ns_loss = HuberLoss(kl_hat, d_nses.detach())
        ns_loss = MSELoss(kl_hat, d_nses.detach())
        self.tran_optimizer.zero_grad()
        ns_loss.backward(retain_graph=True)
        grad_norm = torch.nn.utils.clip_grad_norm(
            self.tran.parameters(), 10 * self.nagents)
        self.tran_optimizer.step()

        if logger is not None:
            logger.add_scalar('losses/ns_loss', ns_loss, self.niter)
            logger.add_scalar('grad_norms/ns', grad_norm, self.niter)
        
        for a_i, pol_loss in enumerate(pol_losses):
            curr_agent = self.agents[a_i]
            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            curr_agent.policy_optimizer.zero_grad()
            pol_loss.backward(retain_graph=True)
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm(
                curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            

            if logger is not None:
                logger.add_scalar('agent%i/losses/pol_loss' % a_i,
                                  pol_loss, self.niter)
                logger.add_scalar('agent%i/grad_norms/pi' % a_i,
                                  grad_norm, self.niter)

        # Update local trust region
        f_loss = sum(pol_losses) + kl_hat.mean()
        for a_i in range(self.nagents):
            disable_gradients(self.policies[a_i])
        self.local_tr_optimizer.zero_grad()
        f_loss.backward()
        for a_i in range(self.nagents):
            enable_gradients(self.policies[a_i])
        grad_norm = torch.nn.utils.clip_grad_norm(self.local_trs, 0.5)
        self.local_tr_optimizer.step()

        if logger is not None:
            logger.add_scalar('losses/f_loss', f_loss, self.niter)
            logger.add_scalar('grad_norms/f', grad_norm, self.niter)

        for a_i, local_tr in enumerate(self.local_trs):
            logger.add_scalar('agent%i/local_tr_clip_before' % \
                a_i, local_tr, self.niter)

        self.trust_region_clipper()
        for a_i, local_tr in enumerate(self.local_trs):
            logger.add_scalar('agent%i/local_tr_clip_after' % \
                a_i, local_tr, self.niter)

    
    def trust_region_clipper(self):
        for a_i in range(self.nagents):
            self.local_trs[a_i].data.clamp_(1e-2, 1e2)


    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        self.tran.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
            a.old_policy.train()
            a.modeling_policy.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.old_pol_dev == device:
            for a in self.agents:
                a.old_policy = fn(a.old_policy)
            self.old_pol_dev = device
        if not self.model_pol_dev == device:
            for a in self.agents:
                a.modeling_policy = fn(a.modeling_policy)
            self.model_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device
        if not self.tran_dev == device:
            self.tran = fn(self.tran)
            self.tran_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4,
                      **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        agent_init_params = []
        sa_size = []
        for acsp, obsp in zip(env.action_space,
                              env.observation_space):
            if len(obsp.shape) > 1:
                agent_init_params.append({'num_in_pol': np.prod(obsp.shape),
                                      'num_out_pol': acsp.n})
                sa_size.append((np.prod(obsp.shape), acsp.n))
            else:
                agent_init_params.append({'num_in_pol': obsp.shape[0],
                                        'num_out_pol': acsp.n})
                sa_size.append((obsp.shape[0], acsp.n))

        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance
