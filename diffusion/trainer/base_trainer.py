import pytorch_lightning as pl

class BaseTrainer(pl.LightningModule):
    def __init__(self, params:dict,train_samples:int, val_samples:int):
        super().__init__()
		
        self.params=params
        self.train_samples=train_samples
        self.val_samples=val_samples