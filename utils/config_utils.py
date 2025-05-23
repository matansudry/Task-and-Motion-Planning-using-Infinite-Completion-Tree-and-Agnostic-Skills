import yaml
from easydict import EasyDict as edict
from datetime import datetime
import os
import json
from utils.general_utils import load_yaml

def update_task(cfg:dict) -> dict:
    pddl_problem = cfg['GENERAL_PARMAS'].pddl_problem
    task_path = pddl_problem.replace("planning/config/problems/", "planning/config/envs/")
    task_path = task_path.replace("pddl", "yaml")
    assert os.path.exists(task_path)
    cfg['GENERAL_PARMAS'].task_path = task_path
    
    return cfg

def load_cfg(config_path:str, load_as_edict:bool=False) -> dict:
    cfg = load_yaml(config_path)

    if load_as_edict:
        cfg = edict(cfg)

    return cfg

def get_time_string(with_milsecond:bool=False) -> str:
    """
    :return: string with the current time in the format 'month_day_hour_minute_second'
    """
    time = datetime.now()
    if with_milsecond:
        return f'{time.month}_{time.day}_{time.hour}_{time.minute}_{time.second}_{time.microsecond}'
    else:
        return f'{time.month}_{time.day}_{time.hour}_{time.minute}_{time.second}'

def edict2dict(edict_obj):
  dict_obj = {}

  for key, vals in edict_obj.items():
    if isinstance(vals, edict):
      dict_obj[key] = edict2dict(vals)
    else:
      dict_obj[key] = vals

  return dict_obj

def save_cfg(cfg, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "config.yml")
    with open(output_path, 'w') as file:
        yaml.dump(edict2dict(cfg), file, indent=4, default_flow_style=False, sort_keys=True)
        
def set_output_path(cfg:dict, output_path:str, make_dir:bool=True, additional_name:str=None) -> dict:
    name = get_time_string(with_milsecond=True)
    if additional_name:
        name = f"{additional_name}_{name}"
    full_path = os.path.join(output_path,name)
    if make_dir:
        os.makedirs(full_path)
    cfg.GENERAL_PARMAS.output_path = full_path

    return cfg

def update_dict(cfg:dict, new_params:dict) -> dict:
    if not isinstance(cfg, dict):
        return new_params
    for key in new_params.keys():
        cfg[key] = update_dict(
            cfg=cfg[key],
            new_params=new_params[key]
        )
    return cfg


def update_cfg_params(cfg:dict, unknown:list) -> dict:
    for param in unknown:
        param = param.replace("--", "")
        key, value = param.split("=")

        #replace ' in "
        value = value.replace("'", "*")
        value = value.replace("\"", "+")
        value = value.replace("+", "'")
        value = value.replace("*", "\"")
        value = json.loads(value)
        new_params = {
            key: value
        }
        cfg = update_dict(cfg=cfg, new_params=new_params)
    print(cfg)
    return cfg