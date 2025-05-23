import sys
sys.path.append(".")

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from lightning_fabric.utilities.seed import seed_everything

from p_estimator.data.data_module import PEstimatorDataModule 
from p_estimator.trainers.trainer import PEstimotarTrainer
from utils.config_utils import load_cfg, set_output_path, save_cfg, get_time_string, update_cfg_params
from p_estimator.callbacks.factory import create_callbacks


def main(cfg:dict):
    run_time = get_time_string()

    #create data module
    data = PEstimatorDataModule(
        cfg=cfg.DATA,
        primitive=cfg.GENERAL_PARMAS.primitive
    )
    data.setup(stage="fit")
    cfg.DATA.train_size = len(data.train_dataset)
    cfg.DATA.val_size = len(data.val_dataset)

    #init wandb
    wandb_logger = WandbLogger(
        project="tamp-p_estimator",
        name=f'{cfg.GENERAL_PARMAS.primitive}_{run_time}',
        config=cfg,
        entity="matansudry"
        )

    #create network
    network = PEstimotarTrainer(config=cfg,)
    
    #create callbacks
    callbacks = create_callbacks(cfg=cfg)
    
    #create trainer
    trainer = pl.Trainer(accelerator="cuda",
                         logger=wandb_logger,
                         log_every_n_steps=1,
                         devices=1,
                         callbacks=callbacks,
                         gradient_clip_algorithm="norm",
                         max_epochs=cfg.TRAINER.TRAINING.num_epochs,
                         gradient_clip_val=2,
                         )

    #train
    trainer.fit(
        model=network,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
        )


if __name__ == '__main__':
    seed_everything(seed=0)

    assert torch.cuda.is_available()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="p_estimator/config/train_config.yml")
    parser.add_argument('--output_path', type=str, default="no_git/checkpoints/p_estimator")
    parser.add_argument('--wandb_off', type=str, default="False")  
    parser.add_argument('--primitive', type=str, default="Pick",\
        choices=["Pick", "Place"])
    args, unknown  = parser.parse_known_args()

    cfg = load_cfg(config_path=args.config_path, load_as_edict=True)
    cfg = update_cfg_params(cfg=cfg, unknown=unknown)
    cfg.GENERAL_PARMAS.log_expirment_wandb = args.wandb_off
    cfg.GENERAL_PARMAS.primitive = args.primitive
    cfg = set_output_path(
        cfg=cfg,
        output_path=args.output_path,
        additional_name=args.primitive
    )


    save_cfg(cfg=cfg, output_folder=cfg.GENERAL_PARMAS.output_path)

    main(cfg=cfg)