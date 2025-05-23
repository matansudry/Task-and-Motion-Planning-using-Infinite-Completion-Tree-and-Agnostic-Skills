import numpy as np
import os
import pickle
import yaml
import json
from easydict import EasyDict as edict
from multiprocessing import Process
from datetime import datetime

from lightning_fabric.utilities.seed import seed_everything


def edict2dict(edict_obj):
  dict_obj = {}

  for key, vals in edict_obj.items():
    if isinstance(vals, edict):
      dict_obj[key] = edict2dict(vals)
    else:
      dict_obj[key] = vals

  return dict_obj

def run_with_limited_time(func, args, kwargs, time):
    """Runs a function with time limit
    :param func: The function to run
    :param args: The functions args, given as tuple
    :param kwargs: The functions keywords, given as dict
    :param time: The time limit in seconds
    :return: True if the function ended successfully. False if it was terminated.
    """
    p = Process(target=func, args=args, kwargs=kwargs)
    p.start()
    p.join(time)
    if p.is_alive():
        p.terminate()
        return False
    return True

def run_with_limited_time_new(func, args=(), kwargs={}, time=1, default=False):
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time)
    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        result = default
    finally:
        signal.alarm(0)

    return result

def get_file_name(number_of_crosses=None, dataset_path="datasets", prefix="")->str:
    """_summary_

    Args:
        number_of_crosses (_type_): _description_
        dataset_path (str, optional): _description_. Defaults to "datasets".
        prefix (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """
    number_of_crosses_str = str(number_of_crosses) if isinstance(number_of_crosses, int) else ""
    folder_path = os.path.join(dataset_path,number_of_crosses_str)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    index = 0
    ok = True
    while(ok):
        file_name = number_of_crosses_str+"_"+prefix+str(index)+".txt"
        if os.path.isfile(os.path.join(folder_path,file_name)):
            index +=1
        else:
            ok = False
    return os.path.join(folder_path,file_name)

def set_seed(seed:int):
    """_summary_

    Args:
        seed (int): seed to set
    """

    seed_everything(seed=seed)

def save_plan(plan:list, main_folder:str, num_of_crosses:int, prefix=""):
    """_summary_

    Args:
        plan (list): _description_
        main_folder (str): _description_
        num_of_crosses (int): _description_
        prefix (str, optional): _description_. Defaults to "".
    """
    name = get_file_name(num_of_crosses,dataset_path=main_folder, prefix=prefix)
    print("sample", name, " was saved")
    with open(name, "wb") as fp:
        pickle.dump(plan, fp)

def save_transition(transition:dict, main_folder:str, prefix=""):
    """_summary_

    Args:
        plan (list): _description_
        main_folder (str): _description_
        num_of_crosses (int): _description_
        prefix (str, optional): _description_. Defaults to "".
    """
    name = get_file_name(dataset_path=main_folder, prefix=prefix)
    with open(name, "wb") as fp:
        pickle.dump(transition, fp)

def save_pickle(path:str, object_to_save:dict):
    """_summary_

    Args:
        path (str): _description_
        object_to_save (dict): _description_
    """
    with open(path, "wb") as fp:
        pickle.dump(object_to_save, fp)
    fp.close()

def save_json(path:str, object_to_save:dict):
    """_summary_

    Args:
        path (str): _description_
        object_to_save (dict): _description_
    """
    with open(path, "w") as fp:
        json.dump(object_to_save, fp)
    fp.close()

def load_pickle(path:str) -> pickle:
    """_summary_

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(path, "rb") as fp:
        file = pickle.load(fp)
    fp.close()
    return file

def save_yaml(path:str, object_to_save):
    with open(path, 'w') as outfile:
        yaml.dump(object_to_save, outfile, default_flow_style=False)

def load_yaml(path:str):
    with open(path, 'r') as fp:
        file = yaml.safe_load(fp)
    fp.close()
    return file

def load_numpy(file_path:str) -> dict:
    raw_data = np.load(file_path, allow_pickle=True)
    sample = raw_data.f.a.tolist()

    return sample

def load_json(file_path:str) -> dict:
    f = open(file_path)
    sample = json.load(f)

    return sample

def convert_action_from_index_to_one_hot_vector(action:np.array, num_of_links:int) -> np.array:
    one_hot = np.zeros(num_of_links+3)
    one_hot[int(action[0])] = 1
    one_hot[num_of_links] = action[1]
    one_hot[num_of_links+1] = action[2]
    one_hot[num_of_links+2] = action[3]

    return one_hot

def get_time_string() -> str:
    """
    :return: string with the current time in the format 'month_day_hour_minute_second'
    """
    time = datetime.now()

    return f'{time.month}_{time.day}_{time.hour}_{time.minute}_{time.second}'