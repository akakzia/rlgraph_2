import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
import  random
from mpi4py import MPI
from arguments import get_args
from utils import get_eval_goals

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

if __name__ == '__main__':
    num_eval = 1
    path = '/home/ahmed/Documents/final_year/ALOE2022/rlgraph/models/'
    model_path = path + 'actor_critic_gn.pt'

    # with open(path + 'config.json', 'r') as f:
    #     params = json.load(f)
    # args = SimpleNamespace(**params)
    args = get_args()

    if args.algo == 'continuous':
        args.env_name = 'FetchManipulate5ObjectsContinuous-v0'
        args.multi_criteria_her = True
    else:
        args.env_name = 'FetchManipulate5Objects-v0'

    # Make the environment
    env = gym.make(args.env_name)

    # set random seeds for reproduce
    args.seed = np.random.randint(1e6)
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    args.env_params = get_env_params(env)

    goal_sampler = GoalSampler(args)

    # create the sac agent to interact with the environment
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward, goal_sampler)
        policy.load(model_path, args)
    else:
        raise NotImplementedError

    # def rollout worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    eval_goals = []
    instructions = ['stack_3', 'stack_4'] * 10
    for instruction in instructions:
        eval_goal = get_eval_goals(instruction, n=args.n_blocks)
        eval_goals.append(eval_goal.squeeze(0))
    eval_goals = np.array(eval_goals)

    all_results = []
    for i in range(num_eval):
        episodes = rollout_worker.generate_rollout(eval_goals, true_eval=True, animated=True)
        results = np.array([e['rewards'][-1] == 5. for e in episodes])
        all_results.append(results)

    results = np.array(all_results)
    print('Av Success Rate: {}'.format(results.mean()))