import sys
sys.path.append(".")
from utils.general_utils import load_yaml

if __name__ == "__main__":
    config = load_yaml("parallel_run/config.yml")
    envs = config['number_of_seeds']
    problems = config['problems']
    max_time = config['max_time']
    name = config['name']
    output_path = f'parallel_run/{name}.txt'
    max_cuda = config['max_cuda']
    f = open(output_path, 'w')
    cuda = 0
    turn_off_high_level = config['turn_off_high_level']
    f.write(f'#nohup cat {output_path} | parallel -j 24 &\n')
    for difficult in config['difficult']:
        problem = problems[difficult]
        if turn_off_high_level:
            for num_of_high_level_plans in config['num_of_high_level_plans']:
                for system in config['systems']:
                    output_path = f'no_git/system/{envs}_{name}_{max_time}'
                    for seed in range(envs):
                        f.write(f'CUDA_VISIBLE_DEVICES={cuda} python planning/main_single_run.py --system {system} --output_path {output_path} --seed {seed} --number_of_high_level_plans {num_of_high_level_plans} --turn_off_high_level True --pddl_problem {problem} \n')
                        cuda +=1 
                        cuda = cuda%max_cuda
                        temp=1
        else:
            for system in config['systems']:
                output_path = f'no_git/system/{system}_{name}_{difficult}_{max_time}_actions_exp_with_first_action_score'
                for seed in range(envs):
                    f.write(f'CUDA_VISIBLE_DEVICES={cuda} python planning/main_single_run.py --system {system} --output_path {output_path} --seed {seed} --pddl_problem {problem} --max_time {max_time} --stochastic_actions\n')
                    cuda +=1 
                    cuda = cuda%max_cuda
                    temp=1
    f.close()