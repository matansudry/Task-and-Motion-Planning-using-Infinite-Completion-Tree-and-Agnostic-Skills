import sys
sys.path.append(".")
from utils.general_utils import load_yaml
import os

def get_config_config(config:dict):
    config["output_path"] = f'parallel_run/{config["name"]}.txt'
    config["stochastic_actions"] = "--stochastic_actions" if config["stochastic_actions"] else ""

    return config

def data_creation(config:dict):
    config = get_config_config(config=config)
    f = open(output_path, 'w')
    cuda = 0
    f.write(f'#nohup cat {output_path} | parallel -j {config["num_of_workers"]} &\n')
    for system in config['systems']:
        for seed in range(seeds[0], seeds[1]):
            for index in range(120):
                problem = os.path.join("planning/config/problems", str(index)+".pddl")
                output_path = f'no_git/system/{name}/{system}_{name}_{index}_{max_time}'
                f.write(f'CUDA_VISIBLE_DEVICES={cuda} nohup python planning/main_single_run.py --system {system} --output_path {output_path} --seed {seed} --pddl_problem {problem} --max_time {max_time} --stochastic_actions\n')
                cuda +=1 
                cuda = cuda%max_cuda
                temp=1
    f.close()

def eval(config:dict):
    config = get_config_config(config=config)
    f = open(config["output_path"], 'w')
    cuda = 0
    f.write(f'#nohup cat {config["output_path"]} | parallel -j {config["num_of_workers"]} &\n')
    for difficult in config['difficult']:
        problem = config["problems"][difficult]
        if config["turn_off_high_level"]:
            raise # Not implemented
            for num_of_high_level_plans in config['num_of_high_level_plans']:
                for system in config['systems']:
                    runs_output_path = f'no_git/system/{envs}_{name}_{max_time}'
                    for seed in range(envs):
                        f.write(f'CUDA_VISIBLE_DEVICES={cuda} python planning/main_single_run.py --system {system} --output_path {runs_output_path} --seed {seed} --number_of_high_level_plans {num_of_high_level_plans} --turn_off_high_level True --pddl_problem {problem} \n')
                        cuda +=1 
                        cuda = cuda%max_cuda
                        temp=1
        else:
            for system in config['systems']:
                state_estimator_options = config['state_estimator'] if system =="tamp" else ["els"]
                for state_estimator in state_estimator_options:
                    runs_output_path = f'no_git/system/{config["name"]}_{system}_{state_estimator}_{difficult}_{config["max_time"]}'
                    for seed in range(config["number_of_seeds"][0], config["number_of_seeds"][1]):
                        f.write(f'CUDA_VISIBLE_DEVICES={cuda} python planning/main_single_run.py --system {system} --output_path {runs_output_path} --seed {seed} --pddl_problem {problem} --state_estimator {state_estimator} --max_time {config["max_time"]} {config["stochastic_actions"]}\n')
                        cuda +=1 
                        cuda = cuda%config["max_cuda"]
                        temp=1
    f.close()


if __name__ == "__main__":
    data_creation_task = False
    config_path = "parallel_run/data_generation_config.yml" if data_creation_task else "parallel_run/eval_config.yml" 
    config = load_yaml(config_path)
    if data_creation_task:
        data_creation(config=config)
    else:
        eval(config=config)
