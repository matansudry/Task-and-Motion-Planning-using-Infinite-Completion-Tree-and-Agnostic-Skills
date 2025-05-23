import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities import grad_norm
import clip

from p_estimator.network.attention import AttentaionNetwork

TYPE =torch.float32

class PEstimotarTrainer(pl.LightningModule):
    def __init__(self, config:dict):
        super().__init__()
        self.network = AttentaionNetwork().to(TYPE)
        self.network.to(self.device)
        self.config=config
        self.loss = nn.MSELoss() #nn.MSELoss() #nn.BCELoss()
        model, preprocess = clip.load("ViT-B/32")
        self.text_embedder = model.eval()
        self.text_embedder.requires_grad_(False)
        self.text_embedder_preprocess = preprocess

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.TRAINER.TRAINING.lr,
        )
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        self.network.train(True)
        prediction, loss = self.run_step(
            batch,
            batch_idx
        )
        self.log("train_loss", loss)
        output = {
            "loss": loss,
            "prediction": prediction,
        }
        return output

    def validation_step(self, batch, batch_idx):
        self.network.train(False)
        prediction, loss = self.run_step(
            batch,
            batch_idx
        )
        self.log("val_loss", loss)
        output = {
            "loss": loss,
            "prediction": prediction,
        }
        return output
    
    def test_step(self, batch, batch_idx):
        self.network.train(False)
        prediction, loss = self.run_step(
            batch,
            batch_idx
        )
        self.log("test_loss", loss)
        output = {
            "loss": loss,
            "prediction": prediction,
        }
        return output

    def run_step(self, batch:dict, batch_idx:int):
        batch["state"] = batch["state"].to(TYPE)
        batch["success_rate"] = batch["success_rate"].to(TYPE)

        text = clip.tokenize(batch['high_level_action']).to(self.device)

        with torch.no_grad():
            text_features = self.text_embedder.encode_text(text).to(TYPE)
        batch["text_features"] = text_features

        output = self.network(batch)
        assert torch.max(output).item() <= 1 and torch.min(output).item() >=0
        loss = self.loss(torch.unsqueeze(batch['success_rate'],1), output)#self.loss(batch['success_rate'], torch.squeeze(output,1))
        return output, loss

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.network, norm_type=2)
        self.log_dict(norms)