from re import S
import torch
import numpy as np
from mpi4py import MPI
import env
import gym
import os
from arguments import get_args
from rl_modules.rl_agent import RLAgent
import random
from rollout import RolloutWorker
from goal_sampler import GoalSampler
from utils import init_storage, get_eval_goals
import time
from mpi_utils import logger

def get_env_params(env):
    obs = env.reset()

    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

def launch(args):
    # Set cuda arguments to True
    args.cuda = torch.cuda.is_available()

    rank = MPI.COMM_WORLD.Get_rank()

    t_total_init = time.time()

    # Make the environment
    if args.algo == 'continuous':
        args.env_name = 'FetchManipulate{}ObjectsContinuous-v0'.format(args.n_blocks)
    else: 
        args.env_name = 'FetchManipulate{}Objects-v0'.format(args.n_blocks)
        args.zpd_management = False
        args.use_curriculum = False
    
    env = gym.make(args.env_name)

    # set random seeds for reproducibility
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    # get saving paths
    if rank == 0:
        logdir, model_path = init_storage(args)
        logger.configure(dir=logdir)
        logger.info(vars(args))

    args.env_params = get_env_params(env)

    goal_sampler = GoalSampler(args)

    # Initialize RL Agent
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward, goal_sampler)
    else:
        raise NotImplementedError

    # Initialize Rollout Worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    # Main interaction loop
    episode_count = 0
    for epoch in range(args.n_epochs):
        t_init = time.time()

        # setup time_tracking
        time_dict = dict(goal_sampler=0,
                         rollout=0,
                         gs_update=0,
                         store=0,
                         norm_update=0,
                         policy_train=0,
                         eval=0,
                         epoch=0)

        # log current epoch
        if rank == 0: logger.info('\n\nEpoch #{}'.format(epoch))

        # Cycles loop
        for _ in range(args.n_cycles):

            # Sample goals
            t_i = time.time()
            goals, self_eval = goal_sampler.sample_goal(n_goals=args.num_rollouts_per_mpi, evaluation=False) # These goals are overridden in rollout_worker
            time_dict['goal_sampler'] += time.time() - t_i

            # Environment interactions
            t_i = time.time()
            episodes = rollout_worker.generate_rollout(goals=goals,  # list of goal configurations
                                                       self_eval=self_eval,
                                                       true_eval=False,  # these are not offline evaluation episodes
                                                       biased_init=args.zpd_management, # Use biased initializations as part of ZPD management scheme
                                                      )
            time_dict['rollout'] += time.time() - t_i

            # Goal Sampler updates
            t_i = time.time()
            if args.use_curriculum or args.algo == 'semantic':
                # Update successes and failures using rollouts aggregated accross workers
                # Or update set of discovered goals for semantic goals
                episodes = goal_sampler.update(episodes)
            time_dict['gs_update'] += time.time() - t_i

            # Storing episodes
            t_i = time.time()
            policy.store(episodes)
            time_dict['store'] += time.time() - t_i

            # Updating observation normalization
            t_i = time.time()
            for e in episodes:
                policy._update_normalizer(e)
            time_dict['norm_update'] += time.time() - t_i

            # Policy updates
            t_i = time.time()
            for _ in range(args.n_batches):
                policy.train()
            time_dict['policy_train'] += time.time() - t_i
            episode_count += args.num_rollouts_per_mpi * args.num_workers

        if args.use_curriculum:
            # Update LP estimation based on goal sampler's successes and failures
            if rank == 0:
                goal_sampler.update_lp()
            goal_sampler.sync()
        time_dict['epoch'] += time.time() -t_init
        time_dict['total'] = time.time() - t_total_init

        if args.evaluations:
            if rank==0: logger.info('\tRunning eval ..')
            # Performing evaluations
            t_i = time.time()
            eval_goals = []
            if args.algo == 'semantic':
                instructions = ['close_1', 'close_2', 'close_3', 'stack_2', 'stack_3', '2stacks_2_2', '2stacks_2_3', 'pyramid_3',
                                'mixed_2_3', 'stack_4', 'stack_5']
                for instruction in instructions:
                    eval_goal = get_eval_goals(instruction, n=args.n_blocks)
                    eval_goals.append(eval_goal.squeeze(0))
                eval_goals = np.array(eval_goals)
            else:
                eval_goals, _ = goal_sampler.sample_goal(evaluation=True)
            episodes = rollout_worker.generate_rollout(goals=eval_goals,
                                                       self_eval=True,
                                                       true_eval=True,  # this is offline evaluations
                                                       biased_init=False
                                                       )

            results = np.array([e['success'][-1].astype(np.float32) for e in episodes])
            rewards = np.array([e['rewards'][-1] for e in episodes])
            all_results = MPI.COMM_WORLD.gather(results, root=0)
            all_rewards = MPI.COMM_WORLD.gather(rewards, root=0)
            time_dict['eval'] += time.time() - t_i

            # Logs
            if rank == 0:
                assert len(all_results) == args.num_workers  # MPI test
                av_res = np.array(all_results).mean(axis=0)
                av_rewards = np.array(all_rewards).mean(axis=0)
                global_sr = np.mean(av_res)
                log_and_save(goal_sampler, epoch, episode_count, av_res, av_rewards, global_sr, time_dict)

                # Saving policy models
                if epoch % args.save_freq == 0:
                    policy.save(model_path, epoch)
                if rank==0: logger.info('\tEpoch #{}: SR: {}'.format(epoch, global_sr))


def log_and_save( goal_sampler, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
    goal_sampler.save(epoch, episode_count, av_res, av_rew, global_sr, time_dict)
    for k, l in goal_sampler.stats.items():
        logger.record_tabular(k, l[-1])
    logger.dump_tabular()


if __name__ == '__main__':
    # Prevent hyperthreading between MPI processes
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    # Get parameters
    args = get_args()

    launch(args)
