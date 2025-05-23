import sys
sys.path.append(".")

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from lightning_fabric.utilities.seed import seed_everything

from utils.config_utils import load_cfg, set_output_path, save_cfg
from utils.config_utils import get_time_string
from p_estimator.data.data_module import PEstimatorDataModule 
from p_estimator.trainers.trainer import PEstimotarTrainer
from p_estimator.callbacks.factory import create_callbacks

def main(cfg:dict):
    run_time = get_time_string()

    #init wandb
    wandb_logger = WandbLogger(
        project="tamp-p_estimator",
        name=f'test_{cfg.GENERAL_PARMAS.primitive}_{run_time}',
        config=cfg,
        entity="matansudry"
        )
    
    #create data module
    data = PEstimatorDataModule(
        cfg=cfg.DATA,
        primitive=cfg.GENERAL_PARMAS.primitive
    )
    data.setup(stage="test")
    #data.setup(stage="fit")

    #create network
    network = PEstimotarTrainer.load_from_checkpoint(
        cfg['GENERAL_PARMAS'].checkpoint_path,
        config=cfg
    )
    
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

    #test
    trainer.test(
        model=network,
        dataloaders=data.test_dataloader(),
        )


if __name__ == '__main__':
    seed_everything(seed=0)

    assert torch.cuda.is_available()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="p_estimator/config/train_config.yml")
    parser.add_argument('--output_path', type=str, default="no_git/checkpoints/p_estimator")
    parser.add_argument('--wandb_off', type=str, default="False")
    parser.add_argument('--checkpoint_path', type=str,\
        default="no_git/checkpoints/p_estimator/golden/Place_12_10_20_23_23_944420/epoch=995-step=182268.ckpt") #"no_git/checkpoints/p_estimator/golden/Pick_12_9_20_24_37_957370/epoch=999-step=279000.ckpt")
    parser.add_argument('--primitive', type=str, default="Place",\
        choices=["Pick", "Place"])
    args = parser.parse_args()

    #assert args.primitive in args.checkpoint_path

    cfg = load_cfg(config_path=args.config_path, load_as_edict=True)
    
    cfg = set_output_path(cfg=cfg, output_path=args.output_path)
    cfg.GENERAL_PARMAS.log_expirment_wandb = args.wandb_off
    cfg.GENERAL_PARMAS.primitive = args.primitive
    cfg.GENERAL_PARMAS.checkpoint_path = args.checkpoint_path

    save_cfg(cfg=cfg, output_folder=cfg.GENERAL_PARMAS.output_path)

    main(cfg=cfg)