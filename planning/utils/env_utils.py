from stap.envs.pybullet.table_env import TableEnv
from utils.general_utils import load_yaml




def primitive_name_from_action(action:str) -> str:
    splits = action.split(" ")
    #removing the (
    primitive = splits[0][1:]
    objects = [splits[1]]
    if len(splits) > 1:
        #removing the )
        objects.append(splits[2][:-1])
    return primitive, objects

def load_env(task_path:str, seed:int):
    env_config = load_yaml(task_path)
    env = TableEnv(**dict(env_config["env_kwargs"]))
    observation, info = env.reset(seed=seed)
    initial_observation = observation
    initial_info = info
    
    return env, initial_observation, initial_info

def fix_high_level_action(high_level_action:str) -> str:
    if high_level_action is None:
        return high_level_action
    #fix high level action
    if "_object" in high_level_action:
        high_level_action = high_level_action.replace("_object", "")
    
    return high_level_action