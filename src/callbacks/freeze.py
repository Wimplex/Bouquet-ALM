from typing import Iterable

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import BaseFinetuning
from torch.optim.optimizer import Optimizer

from src.utils.logger import RankedLogger


log = RankedLogger(name=__name__, rank_zero_only=True)


class FreezeCallback(BaseFinetuning):
    def __init__(self, modules_to_freeze: Iterable[str]):
        super().__init__()

        self._modules_to_freeze = modules_to_freeze

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        for module_name in self._modules_to_freeze:
            for m_name, m in pl_module.named_modules():
                if m_name.startswith(module_name):
                    self.freeze(m)

            log.info(f"Module {module_name} were frozen")

    def finetune_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer) -> None:
        pass

    