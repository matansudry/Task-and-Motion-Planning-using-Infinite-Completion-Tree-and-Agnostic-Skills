import sys
sys.path.append(".")

from utils.general_utils import load_pickle, load_yaml, save_yaml


if __name__ == "__main__":
    options = load_pickle(path="all_options.pickle")
    for index, option in enumerate(options):
        env = load_yaml("planning/config/envs/tamp0_problem_hard_one_tower.yaml")
        new_state = []
        for item in env['env_kwargs']['tasks'][0]['initial_state']:
            if "on(" not in  item or "on(rack" in item:
                new_state.append(item)
        for item in option:
            new_state.append(f'on({item}, {option[item]})')
        env['env_kwargs']['tasks'][0]['initial_state'] = new_state
        save_yaml(path=f'planning/config/envs/{index}.yaml', object_to_save=env)
