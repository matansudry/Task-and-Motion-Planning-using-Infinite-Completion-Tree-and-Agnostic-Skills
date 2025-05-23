import lightning as L
from torch.utils.data import DataLoader
from p_estimator.data.dataset import PEstimatorDataset
from utils.general_utils import load_pickle

class PEstimatorDataModule(L.LightningDataModule):
    def __init__(self, cfg:dict, primitive:str) -> None:
        super().__init__()
        self.cfg=cfg
        self.primitive = primitive
        assert isinstance(self.cfg.SPLIT[self.primitive], list)
        self.split = {
            "train": [],
            "val": [],
            "test": []
        }
        for split in self.cfg.SPLIT[self.primitive]:
            temp_split = load_pickle(split)
            for key in self.split.keys():
                self.split[key].extend(temp_split[key])

    def _prepare_dataset(self, dataset_cfg:dict, split:list):
        dataset_cfg["split"] = split
        return PEstimatorDataset(dataset_cfg=dataset_cfg)

    def setup(self, stage: str):
        if stage == "fit":
            #prepare train and val datasets
            self.train_dataset = self._prepare_dataset(
                dataset_cfg=self.cfg.TRAIN,
                split=self.split["train"],
            )
            print("train dataset length =", len(self.train_dataset))
            self.val_dataset = self._prepare_dataset(
                dataset_cfg=self.cfg.VAL,
                split=self.split["val"],
            )
            print("Val dataset length =", len(self.val_dataset))

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            #prepare test dataset
            self.test_dataset = self._prepare_dataset(
                dataset_cfg=self.cfg.TEST,
                split=self.split["test"],
            )

        if stage == "predict":
            raise
            #prepare predict dataset
            self.predict_dataset = self._prepare_dataset(dataset_cfg=self.cfg.PREDICT)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.TRAIN.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.VAL.num_workers,
       )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.TEST.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)