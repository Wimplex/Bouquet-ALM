from typing import Iterable, Callable

from torch.utils.data import DataLoader, Dataset, ConcatDataset
from pytorch_lightning import LightningDataModule


class A2TDataModule(LightningDataModule):
    def __init__(self, 
                 train_datasets: Iterable[Dataset], 
                 val_datasets: Iterable[Dataset], 
                 train_batch_size: int = 1, 
                 val_batch_size: int = 1, 
                 collate_fn: Callable = None,
                 num_workers: int = 1,
                 pin_memory: bool = False):
        
        super().__init__()

        self.save_hyperparameters(logger=False)

    def setup(self, stage: str = None) -> None:
        if self.trainer is not None:
            if self.hparams.train_batch_size % self.trainer.world_size != 0:
                raise RuntimeError("Batch size is not divisible by the number of devices")
        
            self.per_device_train_batch_size = self.hparams.train_batch_size // self.trainer.world_size
            self.per_device_val_batch_size = self.hparams.val_batch_size // self.trainer.world_size

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=ConcatDataset(self.hparams.train_datasets),
            batch_size=self.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.hparams.collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )
    
    def val_dataloader(self) -> Iterable[DataLoader]:
        return [DataLoader(
            dataset=dataset,
            batch_size=self.per_device_val_batch_size,
            shuffle=False,
            collate_fn=self.hparams.collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        ) for dataset in self.hparams.val_datasets]