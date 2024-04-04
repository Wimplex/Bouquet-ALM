from typing import Iterable, Tuple, Dict, Any

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric


InputBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class Audio2TextExperiment(LightningModule):
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                 frozen_parts: Iterable[str] = None):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def setup(self, stage: str) -> None:
        if self.hparams.frozen_part is not None and stage == "fit":
            for part_name in self.hparams.frozen_parts:
                for p_name, p in self.net.named_parameters():
                    if p_name.startswith(part_name):
                        p.requires_grad_(False)

    def forward(self, mels: torch.Tensor, texts: Iterable[str]):
        return self.net(mels, texts)
    
    def model_step(self, batch: InputBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a single model step with output logits and loss computation.

        :param batch: A tuple of input tensors:
            - Mel-features tensor
            - Input tokens tensor
            - Target tokens tensor

        :return: A tuple of:
            - Loss tensor
            - Output logits tensor
            - Target tokens tensor
        """

        mels, tokens = batch
        input, target = tokens[:-1], tokens[1:]
        logits = self(mels, input)

        # Compute loss ignoring audio part
        logits = logits[..., mels.shape[-1]:]
        loss = self.criterion(logits, target, dim=-1)

        return loss, logits, target
    
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

        # TODO: Нужно ли вообще через trainer стучаться до модели?
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