import numpy as np


def is_success(ag, g):
    return (ag == g).all()


def at_least_one_fallen(observation, n):
    """ Given a observation, returns true if at least one object has fallen """
    dim_body = 10
    dim_object = 15
    obs_objects = np.array([observation[dim_body + dim_object * i: dim_body + dim_object * (i + 1)] for i in range(n)])
    obs_z = obs_objects[:, 2]

    return (obs_z < 0.4).any()


class RolloutWorker:
    def __init__(self, env, policy, goal_sampler, args):

        self.env = env
        self.policy = policy
        self.env_params = args.env_params
        self.goal_sampler = goal_sampler
        self.continuous = args.algo == 'continuous'
        self.args = args

    def generate_rollout(self, goals, self_eval, true_eval, biased_init=False, animated=False):
        # In continuous case, goals correspond to classes of goals (0: no stacks | 1: stack 2 | 2: stack 3 | 3: stack 4 | 4: stack 5)
        episodes = []
        # Reset only once for all the goals in cycle if not performing evaluation
        if not true_eval:
            observation = self.env.unwrapped.reset_goal(goal=np.array(goals[0]), biased_init=biased_init)
        for i in range(goals.shape[0]):
            if true_eval or self_eval:
                observation = self.env.unwrapped.reset_goal(goal=np.array(goals[i]), biased_init=False)
            obs = observation['observation']
            ag = observation['achieved_goal']
            g = observation['desired_goal']

            ep_obs, ep_ag, ep_g, ep_actions, ep_success, ep_rewards = [], [], [], [], [], [],

            # Start to collect samples
            for t in range(self.env_params['max_timesteps']):
                # Run policy for one step
                no_noise = self_eval or true_eval  # do not use exploration noise if running self-evaluations or offline evaluations
                # feed both the observation and mask to the policy module
                action = self.policy.act(obs.copy(), ag.copy(), g.copy(), no_noise)

                # feed the actions into the environment
                if animated:
                    self.env.render()

                observation_new, r, _, info = self.env.step(action)
                obs_new = observation_new['observation']
                ag_new = observation_new['achieved_goal']
                ag_new_bin = observation_new['achieved_goal_binary']
                if self.continuous:
                    success = info['is_success']
                else: 
                    success = is_success(ag_new, g)

                # Append rollouts
                ep_obs.append(obs.copy())
                ep_ag.append(ag.copy())
                ep_g.append(g.copy())
                ep_actions.append(action.copy())
                ep_rewards.append(r)
                ep_success.append(success)

                # Re-assign the observation
                obs = obs_new
                ag = ag_new
                ag_bin = ag_new_bin

            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())

            # Gather everything
            episode = dict(obs=np.array(ep_obs).copy(),
                           act=np.array(ep_actions).copy(),
                           g=np.array(ep_g).copy(),
                           ag=np.array(ep_ag).copy(),
                           success=np.array(ep_success).copy(),
                           rewards=np.array(ep_rewards).copy())
            
            episode['self_eval'] = self_eval
            if self.continuous:
                episode['goal_class'] = goals[i]

            episodes.append(episode)

            # if not eval, make sure that no block has fallen. If so (or success), then reset
            fallen = at_least_one_fallen(obs, self.args.n_blocks)
            if not true_eval and (fallen or success):
                observation = self.env.unwrapped.reset_goal(goal=np.array(goals[i]), biased_init=biased_init)

        return episodes

