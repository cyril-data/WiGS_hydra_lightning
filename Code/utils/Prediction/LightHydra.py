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
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import resource
import logging

import torch.nn as nn
from pathlib import Path

logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
os.environ["PROJECT_ROOT"] = f"{os.getcwd()}/.."


# Remonte jusqu'à ukz42ac/IAlefeu/, puis va dans henrihost-al
HHAL_PATH = Path(__file__).resolve().parents[4] / "henrihost-al"

print("HHAL_PATH", HHAL_PATH)
sys.path.insert(0, str(HHAL_PATH))

from hhal.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def reset_trainer(trainer: Trainer):
    """Remet le Trainer dans un état propre pour un nouveau .fit()"""
    trainer.fit_loop.epoch_progress.reset()
    trainer.fit_loop.epoch_loop.batch_progress.reset()
    trainer.fit_loop.epoch_loop.scheduler_progress.reset()
    trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.reset()

    #     predictor_model.fit_loop.epoch_progress.reset()
    #     predictor_model.fit_loop.epoch_loop.batch_progress.reset()
    #     predictor_model.fit_loop.epoch_loop.scheduler_progress.reset()
    #     predictor_model.fit_loop.epoch_loop.automatic_optimization.optim_progress.reset()

    #         # Vider le cache GPU explicitement
    #         torch.cuda.empty_cache()

    # trainer.fit_loop.epoch_progress.current.completed = 0
    # trainer.fit_loop.epoch_progress.current.processed = 0
    # trainer.fit_loop.epoch_progress.current.started = 0
    # trainer.fit_loop.epoch_progress.current.ready = 0

    # Vider le cache GPU explicitement
    torch.cuda.empty_cache()


def print_gpu_memory():
    # Mémoire totale et libre sur le GPU
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # en GiB
    free_memory = torch.cuda.mem_get_info(0)[0] / (1024**3)  # en GiB
    used_memory = total_memory - free_memory

    # Mémoire allouée par PyTorch
    allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # en GiB
    reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)  # en GiB

    print(f"Mémoire totale GPU : {total_memory:.2f} GiB")
    print(f"Mémoire libre GPU : {free_memory:.2f} GiB")
    print(f"Mémoire utilisée (total) : {used_memory:.2f} GiB")
    print(f"Mémoire allouée par PyTorch : {allocated_memory:.2f} GiB")
    print(f"Mémoire réservée par PyTorch : {reserved_memory:.2f} GiB")
    print("---")


def get_hl_datamodules(cfg: DictConfig) -> LightningDataModule:

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    return datamodule


def get_hl_modules(cfg: DictConfig, datamodule) -> LightningDataModule:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger = True
    logger_cfg = cfg.get("logger", None)
    if logger_cfg and isinstance(logger_cfg, DictConfig):
        logger: List[Logger] = instantiate_loggers(logger_cfg)

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        enable_model_summary=False,
        callbacks=callbacks,
        logger=logger,
    )

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

    return trainer, model


def get_new_model(cfg: DictConfig, datamodule, trainer: Trainer) -> LightningDataModule:

    device = trainer.model.device

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

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

    return model.to(device)


def update_scheduler(trainer, model, datamodule, cfg):

    if cfg["model"].get("scheduler"):
        if "OneCycleLR" in cfg["model"]["scheduler"]["_target_"]:
            steps_per_epoch = int(
                np.ceil(len(datamodule.train_data) / (cfg.batch_size * trainer.world_size))
            )
            print(f"Override steps per epoch to {steps_per_epoch}")
            cfg["model"]["scheduler"]["steps_per_epoch"] = steps_per_epoch

            if hasattr(model, "scheduler"):

                scheduler = model.scheduler
                scheduler.total_steps = cfg.max_epochs * steps_per_epoch

    return trainer, model, datamodule, cfg


def get_hl_cfg(strategy) -> DictConfig:
    with initialize(
        version_base="1.3",
        config_path="../../../../henrihost-al/configs",
    ):
        output_dir = f"{os.environ['PROJECT_ROOT']}/logs/temp"
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=baseline_active_learning",
                # f"csv_path={os.environ['PROJECT_ROOT']}/../henrihost-al/data/database_500k.csv",
                f"paths.output_dir={output_dir}",
                f"paths.work_dir={os.environ['PROJECT_ROOT']}",
            ],
        )
        cfg.logger.mlflow.run_name = f"{cfg.logger.mlflow.run_name}_{strategy}"
        return cfg


def full_datamodule_to_pd(datamodule):
    """Construit df_full depuis le dataset COMPLET, pas depuis train_data."""
    full_X = datamodule.df_x.to_numpy().astype(np.float32)
    full_y_reg = datamodule.df_y_reg.to_numpy().astype(np.float32)
    full_y_time_reg = datamodule.df_y_time_reg.to_numpy().astype(np.float32)
    full_y_cls = datamodule.df_y_cls.to_numpy().astype(np.int32)
    full_y_time_cls = datamodule.df_y_time_cls.to_numpy().astype(np.int32)

    full_y = np.concatenate([full_y_reg, full_y_time_reg, full_y_cls, full_y_time_cls], axis=1)
    y_labels = (
        datamodule.train_data.y_reg_labels
        + datamodule.train_data.y_time_reg_labels
        + datamodule.train_data.y_cls_labels
        + datamodule.train_data.y_time_cls_labels
    )
    full_y_X = np.concatenate([full_y, full_X], axis=1)
    y_X_labels = datamodule.train_data.x_labels + y_labels

    df_full = pd.DataFrame(
        full_y_X, columns=y_X_labels, index=datamodule.df_x.index
    )  # ← index complet
    y_size = len(y_labels)
    return df_full, y_size


# def train_datamodule_to_pd(datamodule):
#     # =============================================================
#     # *** import Train data in pandas frame for active learning ***
#     # train
#     full_X = datamodule.train_data.data_x
#     full_y_reg = datamodule.train_data.data_y_reg
#     full_y_time_reg = datamodule.train_data.data_y_time_reg
#     full_y_cls = datamodule.train_data.data_y_cls
#     full_y_time_cls = datamodule.train_data.data_y_time_cls
#     full_y = np.concatenate([full_y_reg, full_y_time_reg, full_y_cls, full_y_time_cls], axis=1)

#     print("-" * 80)
#     print("train_datamodule_to_pd")
#     print("full_X", full_X.shape)
#     print("full_y", full_y.shape)
#     print("-" * 80)

#     y_labels = (
#         datamodule.train_data.y_reg_labels
#         + datamodule.train_data.y_cls_labels
#         + datamodule.train_data.y_time_reg_labels
#         + datamodule.train_data.y_time_cls_labels
#     )
#     full_y_X = np.concatenate([full_y, full_X], axis=1)
#     y_X_labels = datamodule.train_data.x_labels + y_labels
#     df_full = pd.DataFrame(full_y_X, columns=y_X_labels, index=datamodule.train_data.df_x.index)
#     y_size = len(y_labels)


#     return df_full, y_size
#     # ***


# def val_datamodule_to_pd(datamodule):

#     # *** import Val data in pandas frame for active learning ***
#     # val
#     full_X = datamodule.val_data.data_x
#     full_y_reg = datamodule.val_data.data_y_reg
#     full_y_time_reg = datamodule.val_data.data_y_time_reg
#     full_y_cls = datamodule.val_data.data_y_cls
#     full_y_time_cls = datamodule.val_data.data_y_time_cls
#     full_y = np.concatenate([full_y_reg, full_y_cls, full_y_time_reg, full_y_time_cls], axis=1)

#     print("-" * 80)
#     print("val_datamodule_to_pd")
#     print("full_X", full_X.shape)
#     print("full_y", full_y.shape)
#     print("-" * 80)

#     y_labels = (
#         datamodule.val_data.y_reg_labels
#         + datamodule.val_data.y_cls_labels
#         + datamodule.val_data.y_time_reg_labels
#         + datamodule.val_data.y_time_cls_labels
#     )
#     full_y_X = np.concatenate([full_y, full_X], axis=1)
#     y_X_labels = datamodule.val_data.x_labels + y_labels
#     df_test = pd.DataFrame(full_y_X, columns=y_X_labels, index=datamodule.val_data.df_x.index)
#     y_size = len(y_labels)

#     # ***
#     return df_test, y_size


def hl_pd_to_dataloader(X_candidate, device, dtype):

    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    current = len(os.listdir("/proc/self/fd"))  # Linux seulement

    X_candidate_th = torch.from_numpy(X_candidate.values).to(device)
    X_candidate_th = X_candidate_th.to(dtype=dtype)  # Optionnel

    # dataset = TensorDataset(X_candidate_th)
    dataloader = DataLoader(X_candidate_th, batch_size=4098, shuffle=False)
    return dataloader


def hl_np_to_dataloader(X_candidate, device, dtype):
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    current = len(os.listdir("/proc/self/fd"))  # Linux seulement
    X_candidate_th = torch.from_numpy(X_candidate).to(device)
    X_candidate_th = X_candidate_th.to(dtype=dtype)  # Optionnel

    # dataset = TensorDataset(X_candidate_th)
    dataloader = DataLoader(X_candidate_th, batch_size=4098, shuffle=False)
    return dataloader


def hl_y_pred_pd_to_tensor(y_pred_candidate, all_cols, X_candidate_index):

    all_rows = []
    all_rows_cls = []

    reg_cols = None
    cls_cols = None
    time_reg_cols = None
    time_cls_cols = None

    iteration = 0

    for batch in y_pred_candidate:
        # batch est un dict avec y_reg, y_cls, y_time_reg, y_time_cls

        parts = []
        # parts_cls = []

        # y_reg : [batch, n_reg]
        if batch["y_reg"] is not None:
            parts.append(batch["y_reg"])  # [B, n_reg]
            if reg_cols is None:
                reg_cols = batch["y_reg"].shape[1]

        # y_cls : liste de tenseurs [batch, 2]  → cat sur dim=-1
        if batch["y_cls"] is not None:
            # parts.append(torch.cat(batch["y_cls"], dim=-1))  # [B, K*4]
            # parts_cls.append(batch["y_cls"])  # [B, K*4]

            if cls_cols is None:
                cls_cols = len(batch["y_cls"])

        # y_time_reg : [batch, n_time_reg]
        if batch["y_time_reg"] is not None:
            parts.append(batch["y_time_reg"])  # [B, n_time_reg]
            if time_reg_cols is None:
                time_reg_cols = batch["y_time_reg"].shape[1]

        # y_time_cls : liste de tenseurs [batch, 3] → cat sur dim=-1
        if batch["y_time_cls"] is not None:
            # parts.append(torch.cat(batch["y_time_cls"], dim=-1))  # [B, M*3]
            # parts_cls.append(batch["y_time_cls"])  # [B, M*3]
            if time_cls_cols is None:
                time_cls_cols = len(batch["y_time_cls"])

        row = torch.cat(parts, dim=-1)  # [B, total_cols]
        all_rows.append(row)
        iteration += 1

    # all_rows_cls.append(parts_cls)
    y_pred_candidate_reg = torch.cat(all_rows, dim=0)  # [N_total, total_cols]

    # # --- Définition des colonnes par type ---
    # all_cols = y_train.columns.to_list()

    all_reg_cols = (
        all_cols[0:reg_cols]
        + all_cols[(reg_cols + cls_cols) : (reg_cols + cls_cols + time_reg_cols)]
    )

    y_pred_candidate_pd = pd.DataFrame(
        y_pred_candidate_reg,
        index=X_candidate_index,
        columns=all_reg_cols,  # Nom de la colonne
    )

    return y_pred_candidate_pd, all_reg_cols


def reg_cls_col_selection(
    y_pred_candidate,
    all_cols,
):

    all_rows = []
    all_rows_cls = []

    reg_cols = None
    cls_cols = None
    time_reg_cols = None
    time_cls_cols = None

    parts = []

    batch = y_pred_candidate[0]
    # batch est un dict avec y_reg, y_cls, y_time_reg, y_time_cls

    if batch["y_reg"] is not None:
        reg_cols = batch["y_reg"].shape[1]
        parts.append(batch["y_reg"])  # [B, n_time_reg]

    # y_cls : liste de tenseurs [batch, 2]  → cat sur dim=-1
    if batch["y_cls"] is not None:
        cls_cols = len(batch["y_cls"])

    if batch["y_time_reg"] is not None:
        time_reg_cols = batch["y_time_reg"].shape[1]
        parts.append(batch["y_time_reg"])  # [B, n_time_reg]

    if batch["y_time_cls"] is not None:
        time_cls_cols = len(batch["y_time_cls"])

    row = torch.cat(parts, dim=-1)  # [B, total_cols]

    # all_rows_cls.append(parts_cls)
    y_pred_candidate_reg = torch.cat([row], dim=0)  # [N_total, total_cols]

    all_reg_cols = (
        all_cols[0:reg_cols]
        + all_cols[(reg_cols + cls_cols) : (reg_cols + cls_cols + time_reg_cols)]
    )

    return all_reg_cols


def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()  # Réinitialise avec la méthode par défaut (Kaiming pour Conv2d, Xavier pour Linear)
    elif isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()  # Réinitialise gamma, beta, mean, var
