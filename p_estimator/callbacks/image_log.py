from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tempfile import TemporaryDirectory
import os


from pytorch_lightning import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT


def get_image(prediction:list, gt:list):
    with TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, "image.png")
        fig, ax = plt.subplots(1)
        ax.set_xlabel("prediction - gt")
        x = np.linspace(0, len(prediction)-1, len(prediction))
        y = np.array(prediction) - np.array(gt)
        ax.axis(ymin=-1,ymax=1)
        ax.hlines(0, 0, len(prediction))
        ax.plot(x, y, 'o', color='black')
        fig.savefig(path, format='png')
        ax = plt.gca()
        im_frame = cv2.imread(path)

        return im_frame


class LogImageCallback(Callback):

    def __init__(self):
        self.prediction_dict = {}
        self.gt_dict = {}

    def _batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",\
        outputs: STEP_OUTPUT, batch: Any, batch_idx: int):
        for index, action in enumerate(batch['high_level_action']):
            if action not in self.prediction_dict:
                self.prediction_dict[action] = []
            if action not in self.gt_dict:
                self.gt_dict[action] = []
            self.prediction_dict[action].append(outputs['prediction'][index].item())
            self.gt_dict[action].append(batch['success_rate'][index].item())

    def _epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        for action in self.gt_dict:
            image = get_image(
                prediction=self.prediction_dict[action],
                gt=self.gt_dict[action]
            )
            trainer.logger.log_image(key=f'{action}', images=[image])
        self.prediction_dict = {}
        self.gt_dict = {}

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",\
        outputs: STEP_OUTPUT, batch: Any, batch_idx: int):
        self._batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
        )

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._epoch_end(
            trainer=trainer,
            pl_module=pl_module,
        )

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",\
        outputs: STEP_OUTPUT, batch: Any, batch_idx: int):
        self._batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
        )

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._epoch_end(
            trainer=trainer,
            pl_module=pl_module,
        )