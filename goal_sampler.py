import numpy as np
from utils import get_idxs_per_relation
from mpi4py import MPI


class GoalSampler:
    def __init__(self, args):
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.n_classes = 11 if args.algo == 'semantic' else 5
        self.goal_dim = args.env_params['goal']
        self.algo = args.algo

        if self.algo == 'semantic':
            self.discovered_goals = []
            self.discovered_goals_str = []
            self.use_curriculum = False
        else:
            self.use_curriculum = args.use_curriculum

        if self.use_curriculum:
            self.self_eval_prob = args.self_eval_prob
            self.queue_length = args.queue_length
            self.n_minimum_points = args.n_minimum_points
            self.epsilon_buffer = args.epsilon_buffer

            # LP quantities
            self.successes_and_failures = [[] for _ in range(self.n_classes)]
            self.LP = np.zeros([self.n_classes])
            self.C = np.zeros([self.n_classes])
            self.p = np.ones([self.n_classes]) / self.n_classes

        self.init_stats()

    def sample_goal(self, n_goals=2, evaluation=False):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        if self.algo == 'semantic':
            if evaluation and len(self.discovered_goals) > 0:
                goals = np.random.choice(self.discovered_goals, size=self.num_rollouts_per_mpi)
            else:
                if len(self.discovered_goals) == 0:
                    goals = np.random.choice([-1., 1.], size=(n_goals, self.goal_dim))
                else:
                    # sample uniformly from discovered goals
                    goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
                    goals = np.array(self.discovered_goals)[goal_ids]
            return goals, False
        else:
            if evaluation:
                return np.array([0, 1, 2, 3, 4]), True
            else:
                if self.use_curriculum:
                    # decide whether to self evaluate
                    self_eval = True if np.random.random() < self.self_eval_prob else False
                    if self_eval:
                        # Choose uniformly
                        goals = np.random.choice(range(self.n_classes), size=n_goals)
                    else:
                        # Choose according to LP
                        goals = np.random.choice(range(self.n_classes), p=self.p, size=n_goals)
                else:
                    # Choose one class uniformly (without LP)
                    self_eval = False
                    goals = np.random.choice(range(self.n_classes), size=n_goals)

            return goals, self_eval
    
    def update(self, episodes):
        """ Update the successes and failures """
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

        if self.rank == 0:
            all_episode_list = [e for eps in all_episodes for e in eps]

            if self.algo == 'semantic':
                # update discovered goals
                for e in all_episode_list:
                    # Add last achieved goal to memory if first time encountered
                    if str(e['ag'][-1]) not in self.discovered_goals_str:
                        self.discovered_goals.append(e['ag'][-1].copy())
                        self.discovered_goals_str.append(str(e['ag'][-1]))
            
            else:
                # update successes and failures
                for e in all_episode_list:
                    if e['self_eval']:
                        if (e['rewards'][-1] == 5.):
                            success = 1
                        else:
                            success = 0
                        self.successes_and_failures[e['goal_class']].append(success)
                        # Make sure not to excede queue length
                        if len(self.successes_and_failures[e['goal_class']]) > self.queue_length:
                            self.successes_and_failures = self.successes_and_failures[-self.queue_length:]
        self.sync()

        return episodes
    
    def update_lp(self):
        """ Update C, LP and p """
        if len(self.successes_and_failures) > 0:
            # Compute C and LP per class
            for k in range(self.n_classes):
                n_points = len(self.successes_and_failures[k])
                if n_points > self.n_minimum_points:
                    sf = np.array(self.successes_and_failures[k])
                    self.C[k] = np.mean(sf[n_points // 2:])
                    self.LP[k] = np.abs(np.sum(sf[n_points // 2:]) - np.sum(sf[: n_points // 2])) / n_points
                else: 
                    self.C[k] = 0
                    self.LP[k] = 0
            # Compute p
            if np.sum(self.LP) == 0:
                self.p = np.ones([self.n_classes]) / self.n_classes
            else:
                self.p = self.LP / self.LP.sum()

            if self.p.sum() > 1:
                self.p[np.argmax(self.p)] -= self.p.sum() - 1
            elif self.p.sum() < 1:
                self.p[-1] = 1 - self.p[:-1].sum()
    
    def sync(self):
        """ Synchronize data accross different workers """
        if self.algo == 'semantic':
            # Synchronize set of discovered goals
            self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
            self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)
        else:
            # Synchronize LP estimations
            self.p = MPI.COMM_WORLD.bcast(self.p, root=0)
            self.LP = MPI.COMM_WORLD.bcast(self.LP, root=0)
            self.C = MPI.COMM_WORLD.bcast(self.C, root=0)

    def build_batch(self, batch_size):
        LP = self.LP
        C = self.C
        if np.sum(LP) == 0:
            p = np.ones([self.n_classes]) / self.n_classes
        else:
            p = self.epsilon_buffer *  np.ones([self.n_classes]) / self.n_classes + (1 - self.epsilon_buffer) * LP / LP.sum()
        if p.sum() > 1:
            p[np.argmax(p)] -= p.sum() - 1
        elif p.sum() < 1:
            p[-1] = 1 - p[:-1].sum()

        goal_ids = np.random.choice(range(self.n_classes), p=p, size=batch_size)
        return goal_ids

    def init_stats(self):
        self.stats = dict()
        # Number of classes of eval
        for i in np.arange(1, self.n_classes+1):
            self.stats['Eval_SR_{}'.format(i)] = []
            self.stats['Av_Rew_{}'.format(i)] = []
            if self.use_curriculum:
                self.stats['LP_{}'.format(i)] = []
                self.stats['C_{}'.format(i)] = []
                self.stats['p_{}'.format(i)] = []
        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['global_sr'] = []
        if self.algo == 'semantic':
            self.stats['nb_discovered'] = []
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update',
                'policy_train', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        if self.algo == 'semantic':
            self.stats['nb_discovered'].append(len(self.discovered_goals))
        for g_id in np.arange(1, len(av_res) + 1):
            self.stats['Eval_SR_{}'.format(g_id)].append(av_res[g_id-1])
            self.stats['Av_Rew_{}'.format(g_id)].append(av_rew[g_id-1])
            if self.use_curriculum:
                self.stats['LP_{}'.format(g_id)].append(self.LP[g_id-1])
                self.stats['C_{}'.format(g_id)].append(self.C[g_id-1])
                self.stats['p_{}'.format(g_id)].append(self.p[g_id-1])
