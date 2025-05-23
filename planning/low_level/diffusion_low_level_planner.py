import torch
import numpy as np
import tqdm

from planning.low_level.base_low_level_planner import BaseLowLevelPlanner
from diffusion.networks.gsc.unet_transformer import ScoreNet
from diffusion.networks.gsc.classifier_transformer import ScoreModelMLP
from stap.envs.pybullet.table.primitives import Primitive

from planning.utils.env_utils import primitive_name_from_action
from diffusion.utils.network_utils import load_checkpoint_lightning

from diffusion.networks.catalog import NETWORK_CATALOG

class DiffusionLowLevelPlanner(BaseLowLevelPlanner):
    def __init__(self, params:dict={}):
        super().__init__(params=params)
        
        self._load_models()
        
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
        
        obs0 = torch.tensor(obs0, device=device)
        num_steps = self.params.num_steps
        high_level_action, _  = primitive_name_from_action(action=raw_high_level_action)
        sample_dim = self.params.state_dim + self.params.action_dim + self.params.state_dim
        xt = torch.zeros(self.params.num_samples, sample_dim).to(device)
        observation_indices = torch.Tensor(observation_indices).to(device).unsqueeze(0)
        observation_indices = observation_indices.repeat(self.params.num_samples, 1)
        sde, ones = self.networks[high_level_action].configure_sdes(
            num_steps=num_steps,
            x_T=xt,
            num_samples=self.params.num_samples
        )
        x0 = obs0.view(-1).to(device) #torch.Tensor(np.array(obs0)*2).to(device)
        
        for t in range(num_steps, 0, -1):
            epsilon, alpha_t, alpha_tm1 = sde.sample_epsilon(t * ones, xt, observation_indices)
            pred = (xt - torch.sqrt(1 - alpha_t)*epsilon) / torch.sqrt(alpha_t)
            pred[:, :self.params.state_dim] = x0
            if self.params.use_transition_model:
                raise
                #pred[:, self.params.state_dim+self.params.action_dim:] = \
                    # transition_model(torch.cat(
                        # [pred[:, :self.params.state_dim+self.params.action_dim], 
                        # observation_indices], 
                        # dim=1)
                        # )
            
            pred = torch.clip(pred, -1, 1)
            epsilon = (xt - torch.sqrt(alpha_t)*pred) / torch.sqrt(1 - alpha_t)
            pred_x0 = (xt - torch.sqrt(1 - alpha_t)*epsilon) / torch.sqrt(alpha_t)
            pred_x0 = torch.clip(pred_x0, -1, 1)
            new_epsilon = torch.randn_like(epsilon)
            xt = torch.sqrt(alpha_tm1)*pred_x0 + torch.sqrt(1 - alpha_tm1)*new_epsilon

        xt = xt.detach().cpu().numpy()
        initial_state = xt[:, :self.params.state_dim]
        action = xt[:, self.params.state_dim:self.params.state_dim+self.params.action_dim]
        final_state = xt[:, self.params.state_dim+self.params.action_dim:]
        if classifier is not None:
            scores = classifier(torch.cat(
                [torch.Tensor(initial_state).to(device), observation_indices],
                dim=1
                )).detach().cpu().numpy().squeeze()
            sorted_indices = np.argsort(scores)[::-1]
            xt = xt[sorted_indices]

        initial_state = xt[:, :self.params.state_dim]
        action = xt[:, self.params.state_dim:self.params.state_dim+self.params.action_dim]
        final_state = xt[:, self.params.state_dim+self.params.action_dim:]

        return action[0] #, final_state*0.5, initial_state*0.5    
    