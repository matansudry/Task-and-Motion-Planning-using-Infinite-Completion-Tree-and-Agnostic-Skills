import sys
sys.path.append(".")

import os
import argparse
from typing import Any, Dict, List, Optional, Union
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightning_fabric.utilities.seed import seed_everything

from dataset.catalog import DATASET_CATALOG
from diffusion.trainer.catalog import TRAINER_CATALOG

from utils.config_utils import load_cfg, set_output_path, save_cfg
from utils.config_utils import get_time_string
from diffusion.utils.tensors import numpy_wrap
from stap import agents, envs

#@numpy_wrap
def query_observation_vector(
    policy: agents.RLAgent, observation: torch.Tensor, policy_args: Optional[Any]
) -> torch.Tensor:
    """Numpy wrapper to query the policy actor."""
    return policy.encoder.encode(observation.to(policy.device), policy_args)

def main(cfg:dict):
    run_time = get_time_string()

    #init wandb
    wandb_logger = WandbLogger(
        project="tamp_diffusion",
        name=f'{cfg.GENERAL_PARMAS.primitive}_{run_time}',
        config=cfg,
        entity="matansudry"
        )

    # Load env
    env_kwargs: Dict[str, Any] = {}
    env_kwargs["gui"] = False
    env = envs.load(
        config=cfg.GENERAL_PARMAS.env_config[cfg.GENERAL_PARMAS.primitive],
        **env_kwargs,
        )

    # Load policy.
    policy = agents.load(
        checkpoint=cfg.GENERAL_PARMAS.checkpoint[cfg.GENERAL_PARMAS.primitive],
        env=env,
        env_kwargs={},
        device=cfg.GENERAL_PARMAS.device,
    )
    assert isinstance(policy, agents.RLAgent)
    policy.eval_mode()

    #init train dataset
    train_dataset = DATASET_CATALOG[cfg.DATASET.type](
        params=cfg.DATASET.params.train,
        policy=policy,
        )
    
    #init train dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAINER.params.training.batch_size,
        shuffle=True,
        num_workers=cfg.GENERAL_PARMAS.num_workers,
        )
    
    print("dataset len =", len(train_dataset))


    #init test dataset
    test_dataset = DATASET_CATALOG[cfg.DATASET.type](
        params=cfg.DATASET.params.test,
        policy=policy,
        )

    #init eval dataloader
    eval_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.TRAINER.params.training.batch_size,
        shuffle=False,
        num_workers=cfg.GENERAL_PARMAS.num_workers,
        )   
    
    model = TRAINER_CATALOG[cfg.TRAINER.type](
        params=cfg.TRAINER.params,
        train_samples=len(train_dataloader.dataset),
        val_samples=len(eval_loader.dataset),
        policy=policy
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            save_top_k=3,
            monitor='Validation - loss',
            mode="min",
            dirpath=cfg.GENERAL_PARMAS.output_path
            )
        )

    trainer = pl.Trainer(accelerator=device,
                         logger=wandb_logger,
                         log_every_n_steps=1,
                         callbacks=callbacks,
                         gradient_clip_algorithm="norm",
                         max_epochs=cfg.TRAINER.params.training.num_epochs
                         )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=eval_loader
        )

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    seed_everything(seed=0)

    assert torch.cuda.is_available()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="diffusion/config/config_v0.yml")
    parser.add_argument('--output_path', type=str, default="no_git/checkpoints/diffusion")
    parser.add_argument('--wandb_off', type=str, default="False")  
    parser.add_argument('--primitive', type=str, default="pick",\
        choices=["pick", "place", "pull", "push"])
    args = parser.parse_args()

    cfg = load_cfg(config_path=args.config_path, load_as_edict=True)
    
    cfg = set_output_path(cfg=cfg, output_path=args.output_path)
    cfg.GENERAL_PARMAS.log_expirment_wandb = args.wandb_off
    cfg.GENERAL_PARMAS.primitive = args.primitive
    cfg.DATASET.params.train.primitive = args.primitive
    cfg.DATASET.params.test.primitive = args.primitive

    save_cfg(cfg=cfg, output_folder=cfg.GENERAL_PARMAS.output_path)

    #check the config is running on the same primitive
    assert cfg['DATASET']['params']['train'].primitive.lower() in cfg['GENERAL_PARMAS']['env_config']

    main(cfg=cfg)