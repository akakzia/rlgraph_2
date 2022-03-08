import numpy as np
from utils import get_idxs_per_object


class her_sampler:
    def __init__(self, args, reward_func=None):
        self.reward_type = args.reward_type
        self.replay_strategy = args.replay_strategy
        self.replay_k = args.replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + args.replay_k))
        else:
            self.future_p = 0
        self.multi_criteria_her = args.multi_criteria_her
        if args.algo == 'continuous':
            self.per_object_goal_ids = np.array([np.arange(i * 3, (i + 1) * 3) for i in range(args.n_blocks)])
            self.reward_func = reward_func
        else:
            self.per_object_goal_ids = get_idxs_per_object(n=args.n_blocks)
            self.reward_func = self.compute_reward_semantic
    
    def compute_reward_semantic(self, ag, g, info=None):
        reward = 0.
        for subgoal in self.per_object_goal_ids:
            if (ag[subgoal] == g[subgoal]).all():
                reward = reward + 1.
        return reward

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        if self.multi_criteria_her:
            for sub_goal in self.per_object_goal_ids:
                her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
                future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
                future_offset = future_offset.astype(int)
                future_t = (t_samples + 1 + future_offset)[her_indexes]
                # Replace
                future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
                transition_goals = transitions['g'][her_indexes]
                transition_goals[:, sub_goal] = future_ag[:, sub_goal]
                transitions['g'][her_indexes] = transition_goals
        else:
            # her idx
            her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]

            # replace goal with achieved goal
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            transitions['g'][her_indexes] = future_ag
        transitions['r'] = np.expand_dims(np.array([self.reward_func(ag_next, g, None) for ag_next, g in zip(transitions['ag_next'],
                                                                                            transitions['g'])]), 1)

        return transitions
