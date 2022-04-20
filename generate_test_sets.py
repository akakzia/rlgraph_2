import numpy as np
import pickle as pkl
from utils import get_eval_goals

N_BLOCKS = 5


if __name__ == '__main__':
    save_dir = './test_sets/'
    # Test set 1
    instructions_t1 = ['close_1', 'close_2', 'close_3', 'stack_2', 'stack_3', '2stacks_2_2', '2stacks_2_3', 'pyramid_3',
                               'mixed_2_3', 'stack_4', 'stack_5']
    n_goals_per_class_t1 = 1000

    # Test set 2
    instructions_t2 = ['stack_3', 'pyramid_3']
    n_goals_per_class_t2 = 1000

    # Test set 3
    instructions_t3 = ['2stacks_2_2', '2stacks_2_3', 'mixed_2_3']
    n_goals_per_class_t3 = 1000

    instructions = [instructions_t1, instructions_t2, instructions_t3]
    n_goals = [n_goals_per_class_t1, n_goals_per_class_t2, n_goals_per_class_t3]

    labels = ['test_set_1', 'test_set_2', 'test_set_3']

    for ins, n, label in zip(instructions, n_goals, labels):
        print(f'Generating data for {label:s}')

        eval_goals = []
        for instruction in ins:
            eval_goals_set_per_instruction = set()
            for i in range(n):
                eval_goal = get_eval_goals(instruction, n=N_BLOCKS)
                if label == 'test_set_2':
                    # In the case of stack 3 and pyramids, only store the above predicates part since our function generates specific cases of close
                    eval_goals_set_per_instruction.add(str(eval_goal.squeeze(0)[-20:]))
                else: 
                    eval_goals_set_per_instruction.add(str(eval_goal.squeeze(0)))
            print(f'Held out goals for {instruction:s}: {len(eval_goals_set_per_instruction):d}')
            eval_goals.append(eval_goals_set_per_instruction)
        
        save_file = save_dir + label + '.pkl'
        with open(save_file, 'wb') as f:
            pkl.dump(eval_goals, f)
        print(f'Saving data at {save_file:s}')

        print('=-' * 42)
