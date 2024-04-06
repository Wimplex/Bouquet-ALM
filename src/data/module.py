from typing import Iterable, Union

import hydra
import omegaconf
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from pytorch_lightning import LightningDataModule

from .collate import A2TCollator


def init_datasets(datasets_configs: Iterable[omegaconf.DictConfig], **kwargs) -> Union[Dataset, Iterable[Dataset]]:
    datasets = []
    for dset_cfg in datasets_configs:
        dataset = hydra.utils.instantiate(dset_cfg, **kwargs)
        datasets.append(dataset)
    
    return dataset


class A2TDataModule(LightningDataModule):
    def __init__(self, 
                 train_datasets: Iterable[omegaconf.DictConfig],
                 val_datasets: Iterable[omegaconf.DictConfig],
                 train_batch_size: int = 1, 
                 val_batch_size: int = 1,
                 num_workers: int = 1,
                 pin_memory: bool = False,
                 tokenizer: PreTrainedTokenizer = None):
        
        super().__init__()

        self.save_hyperparameters(logger=False)

        self._tokenizer = tokenizer

    def setup(self, stage: str = None) -> None:
        if self.trainer is not None:
            if self.hparams.train_batch_size % self.trainer.world_size != 0:
                raise RuntimeError("Batch size is not divisible by the number of devices")
        
            self.per_device_train_batch_size = self.hparams.train_batch_size // self.trainer.world_size
            self.per_device_val_batch_size = self.hparams.val_batch_size // self.trainer.world_size

        self.collate_fn = A2TCollator(self._tokenizer.model_max_length, self._tokenizer.pad_token_id)
        self.train_datasets = init_datasets(self.hparams.train_datasets, tokenizer=self._tokenizer)
        self.val_datasets = init_datasets(self.hparams.val_datasets, tokenizer=self._tokenizer)
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=ConcatDataset(self.train_dataset),
            batch_size=self.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )
    
    def val_dataloader(self) -> Iterable[DataLoader]:
        return [DataLoader(
            dataset=dataset,
            batch_size=self.per_device_val_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        ) for dataset in self.val_datasets]