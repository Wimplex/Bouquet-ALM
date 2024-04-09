from typing import Iterable

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.utils.basic import before_task, instantiate_list_configs
from src.utils.logger import RankedLogger


log = RankedLogger(name=__name__, rank_zero_only=True)


def train(cfg: DictConfig) -> None:
    if cfg.get("random_seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating experiment")
    experiment: pl.LightningModule = hydra.utils.instantiate(cfg.experiment)

    log.info(f"Instantiating datamodule")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating callbacks")
    callbacks: Iterable[pl.Callback] = instantiate_list_configs(cfg.callbacks)

    # log.info(f"Instantiating loggers")
    # logger: Iterable[pl.] = hydra.utils.instantiate(cfg.loggers)
    logger = None

    log.info(f"Instantiating trainer")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    trainer.fit(model=experiment, datamodule=datamodule, ckpt_path=cfg.get("cpkt_path"))


@hydra.main("../configs", "train.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    before_task(cfg)
    exit()
    train(cfg)


# TODO:
# 1. Немного переделать класс модели
# 2. Запустить первый эксперимент с обучением проективного слоя


if __name__ == "__main__":
    main()