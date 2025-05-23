import torch
import numpy as np

from diffusion.trainer.base_trainer import BaseTrainer
from diffusion.networks.gsc.unet_transformer import ScoreNet

from diffusion.networks.catalog import NETWORK_CATALOG

from diffusion.utils.network_utils import load_checkpoint_lightning


class DiffusionTrainer(BaseTrainer):
    def __init__(self, params:dict, train_samples:int, val_samples:int, policy):
        super().__init__(
            params=params,
            train_samples=train_samples,
            val_samples=val_samples
        )
        self.policy = policy
        self._load_model()
        self._reset_metrics()

    def _encode(self, observation:torch.tensor, policy_args:dict):
        observation = observation.to(self.device)
        encoded_state = self.policy.encoder.encode(observation, policy_args)
        return encoded_state

    def prepare_batch(self, batch:dict) -> torch.tensor:
        policy_args = {
            "observation_indices" :batch['policy_args']['observation_indices'].cpu().detach().numpy(),
            "shuffle_range" : batch['policy_args']['shuffle_range'].cpu().detach().numpy(),
        }
        start_observation = batch['start_observation']
        start_observation_encod = self.policy.encoder.encode(start_observation, policy_args)
        start_observation_encod = start_observation_encod*2
        
        end_observation = batch['end_observation']
        end_observation_encod = self.policy.encoder.encode(end_observation, policy_args)
        end_observation_encod = end_observation_encod*2 
        
        batch = torch.concat(
            (start_observation_encod.float(),
             batch['action'].float(),
             end_observation_encod.float(),
             batch['obs_indices'].float()),
            dim=1)
        
        return batch

    def _load_model(self):
        """
        load model
        """
        score_model_transition = ScoreNet(
            #num_samples=num_samples,
            #sample_dim=dataset_transition[0].shape[0],
            #condition_dim=0
        )
        self.model = NETWORK_CATALOG[self.params.model.type](
            net=score_model_transition
            )
        

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            raise
        
    def load_checkpoint(self, checkpoint_path:str):
        self.model=load_checkpoint_lightning(
            model=self.model,
            checkpoint_path=checkpoint_path
            )

    def forward(self, batch, cond):
        """_summary_

        Args:
            x (_type_): _description_
            cond (_type_): _description_

        Returns:
            _type_: _description_
        """
        loss = self.model.get_loss(
            x_0=batch,
            obs_ind=cond
            )
        info={}
        return loss.unsqueeze(dim=0), info

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.training.optimizer.lr
            )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.params.training.scheduler.lr_step_size,
            gamma=self.params.training.scheduler.lr_gamma
            )
        return [optimizer], scheduler

    def training_step(self, train_batch, batch_idx):
        train_batch = self.prepare_batch(batch=train_batch)
        data = train_batch[:, :-8]
        cond = train_batch[:, -8:]
        loss, info = self.forward(batch=data, cond=cond)
        
        self.train_metrics["Train - loss"].append(loss.item())
        
        logs = {
            "train_loss": loss.item()
        }

        return {'loss': loss, 'log': logs}

    def _reset_metrics(self):
        self.train_metrics = {
                "Train - loss": [],
            }
        self.val_metrics = {
                "Validation - loss": [],
            }

    def on_train_epoch_end(self):
        metrics = {}
        for key in self.train_metrics:
            metrics[key] = np.mean(np.array(self.train_metrics[key]))
            self.log(key, metrics[key], on_epoch=True, prog_bar=True)

        metrics = {}
        for key in self.val_metrics:
            metrics[key] = np.mean(np.array(self.val_metrics[key])) #self.val_metrics[key] / self.val_samples
            self.log(key, metrics[key], on_epoch=True, prog_bar=True)

        self._reset_metrics()

    def validation_step(self, val_batch, batch_idx):
        val_batch = self.prepare_batch(batch=val_batch)
        data = val_batch[:, :-8]
        cond = val_batch[:, -8:]
        loss, info = self.forward(batch=data, cond=cond)

        self.val_metrics["Validation - loss"].append(loss.item()) # * batch_size

        logs = {
            "val_loss": loss.item()
        }

        return {'log': logs}
