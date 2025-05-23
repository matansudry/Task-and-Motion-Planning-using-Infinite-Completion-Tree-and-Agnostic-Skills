from typing import Any
import torch
import numpy as np

from pytorch_lightning import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT


class MetricCallback(Callback):

    def __init__(self):
        self.error = []

    def batch_end(self, outputs, batch):
        prediction = outputs['prediction'].view(-1)
        success_rate = batch['success_rate']
        error = torch.abs(prediction - success_rate)
        self.error += error.tolist()

    def epoch_end(self, trainer):
        max_value = max(self.error)
        min_value = min(self.error)
        median = np.median(np.array(self.error), axis=0)
        mean = np.mean(np.array(self.error), axis=0)
        
        trainer.logger.log_metrics({"max error": max_value})
        trainer.logger.log_metrics({"min error": min_value})
        trainer.logger.log_metrics({"median error": median})
        trainer.logger.log_metrics({"mean error": mean})

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",\
        outputs: STEP_OUTPUT, batch: Any, batch_idx: int):
        self.batch_end(
            outputs=outputs,
            batch=batch,
        )

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.epoch_end(
            trainer=trainer
            )

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",\
        outputs: STEP_OUTPUT, batch: Any, batch_idx: int):
        self.batch_end(
            outputs=outputs,
            batch=batch,
        )

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.epoch_end(
            trainer=trainer
            )