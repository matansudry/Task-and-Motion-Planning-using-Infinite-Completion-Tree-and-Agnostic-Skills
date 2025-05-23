import torch
import numpy as np
import tqdm
from collections import OrderedDict


from planning.low_level.base_low_level_planner import BaseLowLevelPlanner
from diffusion.networks.gsc.unet_transformer import ScoreNet
from diffusion.networks.gsc.classifier_transformer import ScoreModelMLP

from stap.planners.utils import load
from stap.planners.scod_cem import SCODCEMPlanner
from diffusion.utils.network_utils import load_checkpoint_lightning
from planning.utils.env_utils import primitive_name_from_action, load_env
from stap.envs.pybullet.table.primitives import Primitive

from diffusion.networks.catalog import NETWORK_CATALOG

class STAPLowLevelPlanner(BaseLowLevelPlanner):
    def __init__(self, params:dict={}):
        super().__init__(params=params)
        
        #self._load_models()
        self._load_env(
            task_path=self.params.task_path,
            seed=self.params.seed
        )
        self._init_planner()
        self.policy_order_type_to_number = {
            "pick": 0,
            "place": 1,
            "pull": 2,
            "push": 3,
        }
        
        self.policy_order_number_to_type = {}
        for key in self.policy_order_type_to_number:
            self.policy_order_number_to_type[self.policy_order_type_to_number[key]] = key

        self._validate_order()
        
    def _validate_order(self):
        for index, checkpoint in enumerate(self.params.policy_checkpoints):
            if not self.policy_order_number_to_type[index] in checkpoint:
                raise
        
    def _load_env(self, task_path:str, seed:int):
        env, initial_observation, initial_info = load_env(
            task_path=task_path,
            seed=seed
        )
        print("env loaded")
        self.env = env
        self.initial_observation = initial_observation
        self.initial_info = initial_info
        
    def _init_planner(self):
        self.planner = load(
            config=self.params.config,
            env=self.env,
            policy_checkpoints=self.params.policy_checkpoints,
            scod_checkpoints=self.params.scod_checkpoints,
            dynamics_checkpoint=self.params.dynamics_checkpoint,
            device=self.params.device,
        )
        
    def _load_models(self):
        self.networks = {}
        for action in self.params.checkpoints:
            score_model_transition = ScoreNet()
            model = NETWORK_CATALOG[self.params.network_type](
                net=score_model_transition
                )

            if torch.cuda.is_available():
                model = model.cuda()
            else:
                raise
            
            model=load_checkpoint_lightning(
                model=model,
                checkpoint_path=self.params['checkpoints'][action]
                )
            
            self.networks[action] = model

    def get_action(self, obs0:np.ndarray, observation_indices:list,
            raw_high_level_action:str, device:str, classifier:ScoreModelMLP=None,
            primitive:Primitive=None):
        motion_plan = self.planner.plan(
            observation=obs0,
            action_skeleton=[primitive],
        )
        
        return motion_plan.actions[0]