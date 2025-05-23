import sys
sys.path.append(".")

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Any
import yaml
import ast
import numpy as np
from deepdiff import  DeepDiff
import copy
import pybullet as p
import shutil

from utils.config_utils import load_cfg, get_time_string
from utils.generate_data_utils import generate_symbolic_states, get_state_object_types,\
    get_symbolic_actions, get_states_to_primitives
from stap import envs
from utils.pddl_dataclass import PDDLConfig
from stap.envs.pybullet.table_env import TableEnv
from utils.general_utils import save_pickle, load_pickle
from lightning_fabric.utilities.seed import seed_everything
import tempfile


def update_config_with_primitive(cfg:dict, previous_primitive:str, primitive:str, dataset_path:str):
    assert previous_primitive != primitive
    cfg.env.primitive = primitive
    if previous_primitive is not None:
        upper_previous_primitive = previous_primitive[0].upper() + previous_primitive[1:]
        cfg.general_params.previous_action_folder = os.path.join(dataset_path, upper_previous_primitive)
    
    return cfg
    

def get_index(folder_path:str, suffix:str):
    #get index
    index = 0
    not_found = True
    while not_found:
        temp_path = os.path.join(folder_path, str(index)+suffix)
        if os.path.isfile(temp_path):
            index +=1
            continue
        else:
            not_found = False
    return temp_path

def save_sample(sample:dict, temp_path:str):
    save_pickle(path=temp_path, object_to_save=sample)

def get_env_config(
    states_to_primitives: Dict[str, List[str]],
    primitive: str,
    template_yaml_path: str,
    gui: bool = False,
    seed: int = 0,
    symbolic_action_type: Literal["valid", "invalid"] = "valid",
    save_env_config: bool = True,
    env_config_path: Optional[str] = None,
    env_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct primitive-specific pybullet environment yaml."""
    with open(template_yaml_path, "r") as f:
        env_config = yaml.safe_load(f)

    tasks = []
    for initial_state, primitive_list in states_to_primitives.items():
        tasks.extend(
            [
                {
                    "initial_state": ast.literal_eval(initial_state),
                    "action_skeleton": [p],
                }
                for p in primitive_list
                if primitive in p
            ]
        )

    if symbolic_action_type == "valid":
        # Ensure probabilities of pick(hook) and pick(box) are equal.
        if primitive == "pick":
            num_pick_hook_actions = 0
            for task in tasks:
                if "hook" in task["action_skeleton"][0]:
                    num_pick_hook_actions += 1
            num_pick_box_actions = len(tasks) - num_pick_hook_actions

            for task in tasks:
                if "hook" in task["action_skeleton"][0]:
                    task["prob"] = 0.5 * (1 / num_pick_hook_actions)
                else:
                    task["prob"] = 0.5 * (1 / num_pick_box_actions)

        # Ensure probabilities of place(hook) and place(box) are equal.
        if primitive == "place":
            num_place_hook_actions = 0
            for task in tasks:
                if "hook" in task["action_skeleton"][0]:
                    num_place_hook_actions += 1
            num_place_box_actions = len(tasks) - num_place_hook_actions

            for task in tasks:
                if "hook" in task["action_skeleton"][0]:
                    task["prob"] = 0.5 * (1 / num_place_hook_actions)
                else:
                    task["prob"] = 0.5 * (1 / num_place_box_actions)

    elif symbolic_action_type == "invalid":
        pass

    else:
        raise ValueError(f"Support for {symbolic_action_type} not implemented.")

    env_config["env_kwargs"]["tasks"] = tasks
    env_config["env_kwargs"]["gui"] = gui
    env_config["env_kwargs"]["name"] = (
        f"{primitive}_{seed}" if env_name is None else env_name
    )
    env_config["env_kwargs"]["primitives"] = [primitive]

    if save_env_config:
        if env_config_path is not None:
            if os.path.exists(env_config_path):
                temp=1 #raise ValueError(f"File {env_config_path} already exists.")
            else:
                with open(env_config_path, "w") as f:
                    yaml.safe_dump(env_config, f)
        else:
            raise ValueError("Require environment configuration save path.")

    return env_config

def dumpStateToFile(file_path:str):
    file = open(file_path, "w")
    for i in range(p.getNumBodies()):
        pos, orn = p.getBasePositionAndOrientation(i)
        linVel, angVel = p.getBaseVelocity(i)
        txtPos = "pos=" + str(pos) + "\n"
        txtOrn = "orn=" + str(orn) + "\n"
        txtLinVel = "linVel" + str(linVel) + "\n"
        txtAngVel = "angVel" + str(angVel) + "\n"
        file.write(txtPos)
        file.write(txtOrn)
        file.write(txtLinVel)
        file.write(txtAngVel)
    file.close()

class GenerateDataset():
    def __init__(self, cfg:dict):
        
        self.cfg= copy.deepcopy(cfg)
        self._prepare_data()
        self.load_previous_state()
        self.cnt = 0
        self.files_paths = [1]
        if self.previous_state_load:
            previous_action_folder = self.cfg.general_params.previous_action_folder
            all_sub_folders = os.listdir(previous_action_folder)
            self.files_paths = [os.path.join(
                previous_action_folder,
                sub_folder,
                "sample.pickle"
                ) for sub_folder in all_sub_folders if sub_folder.isnumeric()]
            
        self.verbose = self.cfg.general_params.varbase
        self.primitive = self.get_primitive()
        self.primitive_str = self.primitive.__class__.__name__
        
        #create folder to save sample
        self.cfg.general_params.output_path =\
            os.path.join(self.cfg.general_params.output_path, self.primitive_str)
        os.makedirs(name=self.cfg.general_params.output_path, exist_ok=True)

    def load_previous_state(self):
        #check if need to load file
        self.previous_state_load = True if self.cfg.general_params.previous_action_folder is not None else False
        if self.previous_state_load:
            
            #get previous_primitive from folder
            self.previous_primitive = self.cfg.general_params.previous_action_folder.split("/")[-1]
            
            #update output_path with previousprimitive
            self.cfg.general_params.output_path = os.path.join(
                self.cfg.general_params.output_path, self.previous_primitive
                )
            os.makedirs(self.cfg.general_params.output_path, exist_ok=True)

    def _get_env_config(self, states_to_primitives:dict):    
        self.env_config = get_env_config(
            states_to_primitives=states_to_primitives,
            template_yaml_path=self.cfg.env.template_env_yaml,
            primitive=self.cfg.env.primitive,
            seed=self.cfg.general_params.seed,
            symbolic_action_type=self.cfg.env.symbolic_action_type,
            save_env_config=self.cfg.general_params.save_env_config,
            env_config_path=self.cfg.env.env_config_path,
            env_name=self.cfg.env.name,
        )
        
    def _generate_env(self, env_kwargs:dict={}):
        """_summary_

        Args:
            env_kwargs (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        env_factory = envs.EnvFactory(config=self.env_config)
        env = env_factory(**env_kwargs) #TableEnv
        
        return env
         
    def _prepare_data(self):
        states_to_actions: Dict[str, List[str]] = defaultdict(list)
        all_passable_states = generate_symbolic_states(
            object_types=self.cfg.env.object_types
        )
        for state in all_passable_states:
            state_object_types = get_state_object_types(
                state=state,
                object_types=self.cfg.env.object_types
                )
            actions = get_symbolic_actions(
                state=state,
                object_types=state_object_types,
                pddl_config=self.cfg.env.pddl_config
                )

            if len(actions) > 0:
                states_to_actions[str(state)] = actions
        
        states_to_primitives: Dict[str, List[str]] = get_states_to_primitives(
            states_to_actions=states_to_actions,
            primitive=self.cfg.env.primitive,
        )
        
        self._get_env_config(states_to_primitives=states_to_primitives)
        self.env = self._generate_env()

    def _get_action(self, uniform:bool=False):
        action = self.primitive.sample(uniform=uniform)
        return action

    @staticmethod
    def check_variable_is_list(variable:list):
        if not isinstance(variable, list):
            variable = variable.tolist()
        return variable

    def _get_sample(self, robot_start_state:dict=None, robot_end_state:dict=None,\
        start_observation:np.array=None, end_observation:np.array=None, primitive:str=None,\
        reward:float=None, info:dict=None, action:np.array=None, previous_action:str=None,\
        start_bullet_state_path:str=None, end_bullet_state_path:str=None, task=None, objects:list=None):
        
        action=self.check_variable_is_list(variable=action)
        start_observation=self.check_variable_is_list(variable=start_observation)
        end_observation=self.check_variable_is_list(variable=end_observation)
        
        sample = {
            "robot_start_state": robot_start_state,
            "robot_end_state": robot_end_state,
            "start_observation": start_observation,
            "end_observation": end_observation,
            "primitive": primitive,
            "reward": reward,
            "info": info,
            "action": action,
            "previous_action": previous_action,
            "timestamp": get_time_string(),
            "start_state_path": start_bullet_state_path,
            "end_state_path": end_bullet_state_path,
            "task": task,
            "objects": objects
        }
        
        return sample
    
    def set_state(self, observation:np.array, robot_state:np.array, state_path:str=None):
        """_summary_

        Args:
            observation (np.array): _description_
            robot_state (np.array): _description_
        """
        self.env.robot.gripper.remove_grasp_constraint()
        if state_path is not None:
            p.restoreState(fileName=state_path)
        if isinstance(observation, list):
            observation = np.array(observation)
        self.env.set_observation(observation=observation)
        self.env.robot.set_state(state=robot_state)

    def reset_env(self, task=None, objects=None):
        self.env.reset(task=task, objects=objects)
    
    def _get_state_path(self, folder_path:str):
        path = get_index(folder_path=folder_path, suffix=".bullet")
        return path
    
    def get_state(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        new_state_path = self._get_state_path(folder_path=self.tmpdir)
        if self.previous_state_load:
            temp_file = self.files_paths[self.cnt]
            previous_sample = load_pickle(path=temp_file)
            self.reset_env(objects=previous_sample['objects'])
            previous_state_path = previous_sample['end_state_path']
            state = previous_sample['robot_end_state']
            observation = previous_sample['end_observation']
            self.set_state(
                state_path=previous_state_path,
                observation=observation,
                robot_state=state
                )
            shutil.copyfile(previous_state_path, new_state_path)
        else:
            observation= copy.deepcopy(self.env.get_observation())
            state_id = self.env.get_state()
            state = copy.deepcopy(self.env._states[state_id[0]])
            #dumpStateToFile(file_path=new_state_path)
            p.saveBullet(new_state_path)
        return new_state_path, observation, state

    def step(self, action):
        """_summary_

        Args:
            action (_type_): _description_
        """
        obs, reward, terminated, truncated, info =\
            self.env.step(action, verbose=self.verbose)
        return obs, reward, terminated, truncated, info
    
    def get_primitive(self):
        primitive = self.env.get_primitive()
        return primitive
    
    def get_target_folder(self):
        folders = [os.path.join(self.cfg.general_params.output_path, name)\
            for name in os.listdir(self.cfg.general_params.output_path) if\
            os.path.isdir(os.path.join(self.cfg.general_params.output_path, name))]
        cnt = 1
        found = False
        while not found:
            if os.path.join(self.cfg.general_params.output_path, str(cnt)) in folders:
                cnt+=1
                continue
            else:
                found = True
                folder_path = os.path.join(self.cfg.general_params.output_path, str(cnt))
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def move_states_to_target_folder(self, target_folder:str, start_bullet_state_path:str,\
        end_bullet_state_path:str):
        shutil.move(start_bullet_state_path, os.path.join(target_folder, "start_state.bullet"))
        shutil.move(end_bullet_state_path, os.path.join(target_folder, "end_state.bullet"))
        start_bullet_state_path = os.path.join(target_folder, "start_state.bullet")
        end_bullet_state_path = os.path.join(target_folder, "end_state.bullet")
        
        return start_bullet_state_path, end_bullet_state_path
    
    def run(self, seed:int=None):
        self.tmpdir = None
        self.cnt += 1
        if self.cnt >= len(self.files_paths):
            self.cnt=0
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tmpdir = tmpdir
            if seed is not None:
                seed_everything(seed=seed)

            #reset
            self.reset_env()
            
            #get state
            start_bullet_state_path, start_observation, robot_start_state = self.get_state()
            
            try:
                #get action
                action = self._get_action()
                
                #start video
                self.env.record_start()
                _, reward, _, _, info = self.step(action=action)
                self.env.record_stop()
            except:
                
                return False
            
            #if failed, save in different place the video
            if reward == 0.0:
                full_path = get_index(folder_path="no_git/failuers", suffix=".gif")
                self.env.record_save(
                    path = full_path
                )
                success = False
                return success
            
            task = {}
            arg_objects = [item.name for item in self.env._task.action_skeleton[0].arg_objects]
            task["arg_objects"] = arg_objects
            task["initial_state"] = self.env._task.initial_state
            objects = [key for key in data_generator.env.objects]
            
            
            #get end state
            end_bullet_state_path, end_observation, robot_end_state = self.get_state()
            
            #path the start state
            previous_action = self.files_paths[self.cnt] if self.previous_state_load else None
            
            #get target folder
            target_folder = self.get_target_folder()
            
            #get sample path and save states
            save_sample_path = os.path.join(target_folder, "sample.pickle")
            start_bullet_state_path, end_bullet_state_path = \
                self.move_states_to_target_folder(
                    target_folder=target_folder,
                    start_bullet_state_path=start_bullet_state_path,
                    end_bullet_state_path=end_bullet_state_path,
                    )
            
            #parse sample
            sample = self._get_sample(
                robot_start_state=robot_start_state,
                robot_end_state=robot_end_state,
                start_observation=start_observation,
                end_observation=end_observation,
                primitive=self.primitive_str,
                reward=reward,
                info=info,
                action=action,
                previous_action=previous_action,
                start_bullet_state_path=start_bullet_state_path,
                end_bullet_state_path=end_bullet_state_path,
                task=task,
                objects = objects
            )
            
            #save sample and video
            save_sample(sample=sample, temp_path=save_sample_path)
            self.env.record_save(
                path = save_sample_path.replace("pickle", "gif")
            )
            #raise
            success = True
            return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="configs/generate_dataset_config.yml") #"configs/generate_dataset_config.yml") #
    parser.add_argument('--output_path', type=str, default="no_git/dataset") #"no_git/dataset")
    parser.add_argument('--primitive', type=str, default=None,\
        choices=["pick", "place", "pull", "push"])
    parser.add_argument('--previous_primitive', type=str, default=None,\
        choices=["pick", "place", "pull", "push"])
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--number_of_samples', type=int, default=10)
    
    
    args = parser.parse_args()
    
    cfg = load_cfg(config_path=args.config_path, load_as_edict=True)
    cfg.general_params.output_path = os.path.join(args.output_path)
    cfg.general_params.seed = args.seed
    if args.primitive is not None:
        cfg.env.primitive = args.primitive
    """
    from utils.config_utils import save_cfg
    save_cfg(
        cfg=cfg,
        output_folder=cfg.general_params.output_path
    )
    """
    cfg.env.pddl_config = PDDLConfig()
    
    cfg = update_config_with_primitive(
        cfg=cfg,
        previous_primitive=args.previous_primitive,
        primitive = args.primitive,
        dataset_path=args.output_path
    )
    

    cnt = 0
    while cnt < args.number_of_samples:
        try:
            data_generator = GenerateDataset(
                cfg=cfg
            )
            success = data_generator.run(seed=cfg.general_params.seed)
            cfg.general_params.seed += 1
            if success:
                cnt +=1
        except:
            cfg.general_params.seed += 1
    
    
    
    