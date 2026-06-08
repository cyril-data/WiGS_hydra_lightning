# ***
# Need to git clone https://gitlab.inria.fr/gschmit/henrihost-al  branch sisr
#     and `pip install -e .` to be able to import light hydra module for IAlefeu
# ***

import os

import sys
from typing import Any, Dict, List, Optional, Tuple
from hydra import initialize_config_dir, compose, initialize

import hydra
import lightning as L
import numpy as np
import rootutils
import torch
import functools
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
import torch.multiprocessing as mp
import yaml
from omegaconf import OmegaConf, DictConfig

os.environ["PROJECT_ROOT"] = f"{os.getcwd()}/.."

from hhal.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def get_hl_modules(cfg: DictConfig) -> LightningDataModule:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # log.info("Instantiating callbacks...")
    # callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # log.info("Instantiating loggers...")
    # logger = None
    # logger_cfg = cfg.get("logger", None)
    # if logger_cfg and isinstance(logger_cfg, DictConfig):
    #     logger: List[Logger] = instantiate_loggers(logger_cfg)

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
    # , callbacks=callbacks, logger=logger)

    if cfg.get("model"):
        # Update model head sizes based on dataset lengths
        cfg["model"]["net"]["input_size"] = datamodule.train_data.data_x.shape[-1]
        cfg["model"]["net"]["num_reg"] = (
            datamodule.train_data.data_y_reg.shape[-1] if cfg["enable_regression"] else 0
        )
        cfg["model"]["net"]["num_time_reg"] = (
            datamodule.train_data.data_y_time_reg.shape[-1] if cfg["enable_time_regression"] else 0
        )
        cfg["model"]["net"]["num_cls"] = (
            datamodule.train_data.data_y_cls.shape[-1] if cfg["enable_cls"] else 0
        )
        cfg["model"]["net"]["num_time_cls"] = (
            datamodule.train_data.data_y_time_cls.shape[-1] if cfg["enable_time_cls"] else 0
        )

        # Update scheduler steps per epoch
        if cfg["model"].get("scheduler"):
            if "OneCycleLR" in cfg["model"]["scheduler"]["_target_"]:
                steps_per_epoch = int(
                    np.ceil(len(datamodule.train_data) / (cfg.batch_size * trainer.world_size))
                )
                print(f"Override steps per epoch to {steps_per_epoch}")
                print("Computing steps per epoch for scheduler")
                cfg["model"]["scheduler"]["steps_per_epoch"] = steps_per_epoch

    model: LightningModule = hydra.utils.instantiate(cfg.model)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        # "callbacks": callbacks,
        # "logger": logger,
        "trainer": trainer,
    }
    # for cb in callbacks:
    #     if hasattr(cb, "set_object_dict"):
    #         image_log_dir = os.path.join(cfg.paths.output_dir, "checkpoints")
    #         os.makedirs(image_log_dir, exist_ok=True)
    #         cb.set_object_dict(object_dict)
    #         cb.set_path_dir(image_log_dir)
    # if logger:
    #     log.info("Logging hyperparameters!")
    #     log_hyperparameters(object_dict)

    return datamodule, trainer


def get_hl_cfg() -> DictConfig:
    with initialize(
        version_base="1.3",
        config_path="../../../../henrihost-al/configs",
    ):
        output_dir = f"{os.environ['PROJECT_ROOT']}/logs/temp"
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=baseline",
                f"csv_path={os.environ['PROJECT_ROOT']}/../henrihost-al/data/database_1M.csv",
                f"paths.output_dir={output_dir}",
                f"paths.work_dir={os.environ['PROJECT_ROOT']}",
            ],
        )
        return cfg


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger = None
    logger_cfg = cfg.get("logger", None)
    if logger_cfg and isinstance(logger_cfg, DictConfig):
        logger: List[Logger] = instantiate_loggers(logger_cfg)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if cfg.get("model"):
        # Update model head sizes based on dataset lengths
        cfg["model"]["net"]["input_size"] = datamodule.train_data.data_x.shape[-1]
        cfg["model"]["net"]["num_reg"] = (
            datamodule.train_data.data_y_reg.shape[-1] if cfg["enable_regression"] else 0
        )
        cfg["model"]["net"]["num_time_reg"] = (
            datamodule.train_data.data_y_time_reg.shape[-1] if cfg["enable_time_regression"] else 0
        )
        cfg["model"]["net"]["num_cls"] = (
            datamodule.train_data.data_y_cls.shape[-1] if cfg["enable_cls"] else 0
        )
        cfg["model"]["net"]["num_time_cls"] = (
            datamodule.train_data.data_y_time_cls.shape[-1] if cfg["enable_time_cls"] else 0
        )

        # Update scheduler steps per epoch
        if cfg["model"].get("scheduler"):
            if "OneCycleLR" in cfg["model"]["scheduler"]["_target_"]:
                steps_per_epoch = int(
                    np.ceil(len(datamodule.train_data) / (cfg.batch_size * trainer.world_size))
                )
                print(f"Override steps per epoch to {steps_per_epoch}")
                print("Computing steps per epoch for scheduler")
                cfg["model"]["scheduler"]["steps_per_epoch"] = steps_per_epoch

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    for cb in callbacks:
        if hasattr(cb, "set_object_dict"):
            image_log_dir = os.path.join(cfg.paths.output_dir, "checkpoints")
            os.makedirs(image_log_dir, exist_ok=True)
            cb.set_object_dict(object_dict)
            cb.set_path_dir(image_log_dir)
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # ckpt_path = cfg.get("ckpt_path", None)
    # if ckpt_path:
    #     model_class = get_class(cfg.model._target_)
    #     model = model_class.load_from_checkpoint(ckpt_path)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
        log.info(f"Best ckpt path: {ckpt_path}")

    train_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics}

    return metric_dict, object_dict, trainer


@hydra.main(
    version_base="1.3",
    config_path="/home/cregan/documents/code/00_LORIA/IAlefeu/henrihost-al/configs",
    config_name="train.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with metric dictionnary.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)

    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # return optimized metric
    return metric_dict


if __name__ == "__main__":
    main()
