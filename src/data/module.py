from typing import Iterable, Union
from functools import partial

from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from pytorch_lightning import LightningDataModule

from .collate import A2TCollator
from src.utils.logger import RankedLogger


log = RankedLogger(name=__name__, rank_zero_only=True)


def instantiate_partial_datasets(partial_datasets: Union[partial, Iterable[partial]], 
                                 **kwargs) -> Union[Dataset, Iterable[Dataset]]:
    if not isinstance(partial_datasets, Iterable):
        partial_datasets = [partial_datasets]
    datasets = [part_dset(**kwargs) for part_dset in partial_datasets]
    
    return datasets


class A2TDataModule(LightningDataModule):
    def __init__(self, 
                 train_datasets: Union[partial, Iterable[partial]],
                 val_datasets: Union[partial, Iterable[partial]],
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
        
        log.info(f"Instantiating train datasets")
        self.train_datasets = instantiate_partial_datasets(
            self.hparams.train_datasets, tokenizer=self._tokenizer)

        log.info(f"Instantiating val datasets")
        self.val_datasets = instantiate_partial_datasets(
            self.hparams.val_datasets, tokenizer=self._tokenizer)
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=ConcatDataset(self.train_dataset),
            batch_size=self.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True
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