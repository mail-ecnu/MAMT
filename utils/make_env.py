import numpy as np
from gym import spaces

"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

class pursuit_parallel_wrapper():
    def __init__(self, parallel_env):
        self.env = parallel_env
        self.agents = sorted(parallel_env.action_spaces.keys())
        self.action_space = [parallel_env.action_spaces[agent] for agent in self.agents]
        self.observation_space = [parallel_env.observation_spaces[agent] for agent in self.agents]

    def reset(self):
        obs = self.env.reset()
        obs_n = [obs[agent].flatten() for agent in self.agents]
        return obs_n

    def seed(self, seed):
        return self.env.seed(seed)

    def step(self, action_n):
        action_n = np.argmax(action_n, axis=1)
        actions = {agent:action_n[i] for i, agent in enumerate(self.agents)}
        obs, rews, dones, infos = self.env.step(actions)
        obs_n = [obs[agent].flatten() for agent in self.agents]
        reward_n = [rews[agent] for agent in self.agents]
        done_n = [dones[agent] for agent in self.agents]
        info_n = {'n': [infos[agent] for agent in self.agents]}
        
        return obs_n, reward_n, done_n, info_n

class walker_parallel_wrapper():
    def __init__(self, parallel_env):
        self.env = parallel_env
        self.agents = sorted(parallel_env.action_spaces.keys())
        self.action_space = [spaces.Discrete(16) for agent in self.agents]
        self.observation_space = [parallel_env.observation_spaces[agent] for agent in self.agents]
        self.old_action_n = None
        self.delta_a = 0.2
        self.action_map = np.array([[int(bit) for bit in f'{i:04b}'] for i in range(16)])
        self.action_map[self.action_map == 0] = -1
        self.action_map = self.action_map * self.delta_a

    def reset(self):
        obs = self.env.reset()
        obs_n = [obs[agent].flatten() for agent in self.agents]
        return obs_n

    def seed(self, seed):
        return self.env.seed(seed)

    def step(self, action_n):
        action_n = np.matmul(action_n, self.action_map) 
        action_n = action_n + self.old_action_n if (self.old_action_n is not None) else action_n
        action_n = np.clip(action_n, -1.0, 1.0)
        actions = {agent:action_n[i] for i, agent in enumerate(self.agents)}
        obs, rews, dones, infos = self.env.step(actions)
        obs_n = [obs[agent].flatten() for agent in self.agents]
        reward_n = [rews[agent] for agent in self.agents]
        done_n = [dones[agent] for agent in self.agents]
        info_n = {'n': [infos[agent] for agent in self.agents]}

        self.old_action_n = action_n
        
        return obs_n, reward_n, done_n, info_n


def make_env(scenario_name, benchmark=False, discrete_action=False, pettingzoo=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    if not pettingzoo:
        from multiagent.environment import MultiAgentEnv
        import multiagent.scenarios as old_scenarios
        import envs.mpe_scenarios as new_scenarios

        # load scenario from script
        try:
            scenario = old_scenarios.load(scenario_name + ".py").Scenario()
        except:
            scenario = new_scenarios.load(scenario_name + ".py").Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        if hasattr(scenario, 'post_step'):
            post_step = scenario.post_step
        else:
            post_step = None
        if benchmark:        
            env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                                reward_callback=scenario.reward,
                                observation_callback=scenario.observation,
                                post_step_callback=post_step,
                                info_callback=scenario.benchmark_data,
                                discrete_action=discrete_action)
        else:
            env = MultiAgentEnv(world, reset_callback=scenario.reset_world,
                                reward_callback=scenario.reward,
                                observation_callback=scenario.observation,
                                post_step_callback=post_step,
                                discrete_action=discrete_action)
        return env
    else:
        if scenario_name == 'pursuit':
            from pettingzoo.sisl import pursuit_v2
            env = pursuit_parallel_wrapper(pursuit_v2.parallel_env())
            return env
        elif scenario_name == 'walker':
            from pettingzoo.sisl import multiwalker_v5
            env = walker_parallel_wrapper(multiwalker_v5.parallel_env())
            return env


