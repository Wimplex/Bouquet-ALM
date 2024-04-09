from typing import Tuple, Dict, Any

import torch
from einops import rearrange
from torchmetrics import MeanMetric
from pytorch_lightning import LightningModule

from src.utils.logger import RankedLogger


InputBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
log = RankedLogger(name=__name__, rank_zero_only=True)


class Audio2TextExperiment(LightningModule):
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, 
                mels: torch.Tensor, 
                tokens: torch.Tensor, 
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.net(mels, tokens, attention_mask)
    
    def model_step(self, batch: InputBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a single model step with output logits and loss computation.

        :param batch: A tuple of input tensors:
            - Mel-features tensor
            - Input tokens tensor
            - Attention mask tensor

        :return: A tuple of:
            - Loss tensor
            - Output logits tensor
            - Target tokens tensor
        """
        mels, tokens, attn_mask = batch
        inputs, targets = tokens[:,:-1], tokens[:,1:]
        attn_mask = attn_mask[...,:-1]
        output = self(mels, inputs, attn_mask)

        # Compute loss only for text instruct
        logits = output["logits"][:, -inputs.shape[-1]:]
        loss = self.criterion(
            rearrange(logits, "b t c -> (b t) c"), 
            rearrange(targets, "b t -> (b t)")
        )

        return loss, logits, targets
    
    def training_step(self, batch: InputBatch, batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.log("train/curr_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch: InputBatch, batch_idx: int, dataloader_idx: int = 0) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)

    def test_step(self, batch: InputBatch, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)

    def on_train_epoch_end(self) -> None:
        mean_loss = self.train_loss.compute()
        self.log("train/loss", mean_loss, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val/loss", self.val_loss.compute(), on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        self.log("test/loss", self.test_loss.compute(), on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
    
        return {"optimizer": optimizer}