import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC


def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    # if not model_dir.exists():
    #     run_num = 1
    # else:
    #     exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
    #                      model_dir.iterdir() if
    #                      str(folder.name).startswith('run')]
    #     if len(exst_run_nums) == 0:
    #         run_num = 1
    #     else:
    #         run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % config.run_num
    run_dir = model_dir / curr_run
    # log_dir = run_dir / 'logs'
    # os.makedirs(log_dir)
    # logger = SummaryWriter(str(log_dir))

    # torch.manual_seed(run_num)
    # np.random.seed(run_num)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    # env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed)
    if config.run_num < 0:
        model = AttentionSAC.init_from_env(env,
                                        tau=config.tau,
                                        pi_lr=config.pi_lr,
                                        q_lr=config.q_lr,
                                        gamma=config.gamma,
                                        pol_hidden_dim=config.pol_hidden_dim,
                                        critic_hidden_dim=config.critic_hidden_dim,
                                        attend_heads=config.attend_heads,
                                        reward_scale=config.reward_scale)
    else:
        model = AttentionSAC.init_from_save(run_dir / 'model.pt', load_critic=True)
        print(f'Successfully loaded model from {run_dir}!')
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        model.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]
            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
    
    assert len(replay_buffer) == config.buffer_length
    if config.run_num < 0:
        dataset_dir = Path('data') / config.env_id / config.model_name / 'random' / str(config.seed)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / 'random_states'
    else:
        dataset_dir = Path('data') / config.env_id / config.model_name / 'elite' / str(config.seed)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / 'elite_states'
    replay_buffer.to_file(dataset_path)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--run_num", default=-1, type=int)
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    parser.add_argument("--n_episodes", default=120, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')

    config = parser.parse_args()

    assert config.n_episodes % config.n_rollout_threads == 0

    config.buffer_length = config.n_rollout_threads * config.episode_length \
        * (config.n_episodes // config.n_rollout_threads)

    run(config)
