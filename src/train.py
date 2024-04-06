import hydra
import omegaconf
import pytorch_lightning as pl

from src.utils.basic import extras
from src.utils.logger import RankedLogger


log = RankedLogger(name=__name__, rank_zero_only=True)


def train(cfg: omegaconf.DictConfig) -> None:
    omegaconf.OmegaConf.resolve(cfg)

    if cfg.get("random_seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating experiment")
    experiment: pl.LightningModule = hydra.utils.instantiate(cfg.experiment)

    log.info(f"Instantiating datamodule")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)

    # log.info(f"Instantiating callbacks")
    # callbacks: Iterable[pl.Callback] = hydra.utils.instantiate(cfg.callbacks)
    callbacks = None

    # log.info(f"Instantiating loggers")
    # logger: Iterable[pl.] = hydra.utils.instantiate(cfg.loggers)
    logger = None

    log.info(f"Instantiate trainer")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    trainer.fit(model=experiment, datamodule=datamodule, ckpt_path=cfg.get("cpkt_path"))


@hydra.main("../configs", "train.yaml", version_base="1.1")
def main(cfg: omegaconf.DictConfig) -> None:
    extras(cfg)
    train(cfg)


# TODO:
# 1. Немного переделать класс модели
# 2. Создать и отладить дерево конфигов
# 3. Запустить первый эксперимент


# def main():

#     model_type = "camellia"
#     model = ALM(ALM_SETTINGS[model_type])
    
#     # Forward model
#     bsize = 2
#     mels = torch.randn([bsize, 80, 3000])
#     texts = "First sentence, here we go!<|batch_sep|>A-a-and the second one!"
#     out = model(mels, texts)
#     print(out.__dict__.keys())


if __name__ == "__main__":
    main()