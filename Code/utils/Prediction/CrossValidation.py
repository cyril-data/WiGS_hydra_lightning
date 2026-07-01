import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim

from utils.Prediction.LightHydra import (
    get_hl_cfg,
    get_hl_modules,
    get_hl_datamodules,
    update_scheduler,
    reset_trainer,
    get_new_model,
)

import gc
import copy
from utils.Prediction.LightHydra import print_gpu_memory, reset_weights
import time

import torch.multiprocessing as mp
from queue import Empty
import pytorch_lightning as pl
from omegaconf import OmegaConf

import os


def get_tensors_snapshot():
    """Retourne un set des ids de tensors CUDA actuellement en vie"""
    gc.collect()
    tensors = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensors[id(obj)] = (
                    obj.shape,
                    obj.dtype,
                    round(obj.element_size() * obj.nelement() / 1e6, 3),
                )
        except:
            pass
    return tensors


def get_cv_rmse(model_object, X_train, y_train, k=5):
    """
    Calculates the K-fold CV RMSE for the current training set.

    Args:
        model_object: The scikit-learn compatible model object
                      (e.g., the .model inside your RidgeRegressionPredictor).
        X_train (pd.DataFrame): Current training features.
        y_train (pd.Series): Current training labels.
        k (int): Number of folds.
    """

    ### Set up ###
    if len(y_train) < k * 2:
        return np.nan
    model_to_cv = model_object

    ### Define the scoring ###
    scores = cross_val_score(
        model_to_cv,
        X_train,
        y_train,
        cv=KFold(n_splits=k, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
    )

    return -np.mean(scores)


def get_cv_rmse_hl(predictor_model, hl_model, datamodule, hl_cfg):
    saved_train_labels = list(datamodule.train_data.labels)  # ← labels pandas
    saved_val_labels = list(datamodule.val_data.labels)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    saved_state_dict = copy.deepcopy(hl_model.state_dict())

    fold_idx = 0

    for cv_train_pos, cv_val_pos in kf.split(saved_train_labels):
        cv_train_labels = [saved_train_labels[i] for i in cv_train_pos]
        cv_val_labels = [saved_train_labels[i] for i in cv_val_pos]

        datamodule.train_data.update_indices(cv_train_labels)
        datamodule.val_data.update_indices(cv_val_labels)

        reset_trainer(predictor_model)
        predictor_model, hl_model, datamodule, hl_cfg = update_scheduler(
            predictor_model, hl_model, datamodule, hl_cfg
        )

        hl_model.apply(reset_weights)

        StartTime = time.time()
        predictor_model.fit(model=hl_model, datamodule=datamodule, ckpt_path=None)
        print(f"\t+++ CV fold {fold_idx} training : {time.time() - StartTime} +++")

        metric_dict = {**predictor_model.callback_metrics}
        rse = metric_dict["val/reg_loss"] + metric_dict["val/reg_time_loss"]
        scores.append(-torch.sqrt(rse).numpy())

        fold_idx += 1

    # Restaurer
    datamodule.train_data.update_indices(saved_train_labels)
    datamodule.val_data.update_indices(saved_val_labels)
    hl_model.load_state_dict(saved_state_dict)

    return -np.mean(scores)


def get_cv_rmse_NN(model_object, X_train, y_train, k=5, epochs=50, batch_size=32, lr=0.001):
    """
    Calculates the K-fold CV RMSE for a PyTorch model.

    Args:
        model_object: The PyTorch model (e.g., MLPredictor instance).
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        k (int): Number of folds.
        epochs (int): Number of training epochs per fold.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.

    Returns:
        float: Mean RMSE across all folds.
    """
    if len(y_train) < k * 2:
        return np.nan

    # Convert to numpy if DataFrame/Series
    if hasattr(X_train, "values"):
        X_train = X_train.values.astype(np.float32)
    if hasattr(y_train, "values"):
        y_train = y_train.values.astype(np.float32).reshape(-1, 1)

    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmse_scores = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for train_idx, val_idx in kf.split(X_train):
        # Split data
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.float32).to(device)

        # Create a fresh model instance for each fold
        model = model_object.__class__(
            input_size=model_object.input_size,
            hidden_size=model_object.hidden_size,
            output_size=model_object.output_size,
            lr=lr,
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor)
            mse = criterion(val_preds, y_val_tensor)
            rmse = torch.sqrt(mse).item()
            rmse_scores.append(rmse)

    return np.mean(rmse_scores)


# def _cv_worker(gpu_id, task_queue, result_queue, hl_cfg_dict, dataloader_workers):

#     hl_cfg = OmegaConf.create(hl_cfg_dict)
#     hl_cfg.trainer.devices = [gpu_id]
#     hl_cfg.trainer.accelerator = "gpu"
#     hl_cfg.data.num_workers = dataloader_workers

#     datamodule = get_hl_datamodules(hl_cfg)
#     hl_trainer, hl_model = get_hl_modules(hl_cfg, datamodule)  # créé UNE SEULE FOIS
#     saved_state_dict = copy.deepcopy(hl_model.state_dict())

#     while True:
#         try:
#             fold_idx, cv_train_labels, cv_val_labels = task_queue.get_nowait()
#         except Empty:
#             break

#         datamodule.train_data.update_indices(cv_train_labels)
#         datamodule.val_data.update_indices(cv_val_labels)

#         reset_trainer(hl_trainer)  # comme dans ton flux principal
#         hl_model.load_state_dict(saved_state_dict)
#         hl_model.apply(reset_weights)


def _cv_worker(gpu_id, task_queue, result_queue, hl_cfg_dict, dataloader_workers):

    print("_cv_worker gpu_id:", gpu_id, "num_workers:", dataloader_workers)
    hl_cfg = OmegaConf.create(hl_cfg_dict)
    hl_cfg.trainer.devices = [gpu_id]
    hl_cfg.trainer.accelerator = "gpu"
    hl_cfg.data.num_workers = dataloader_workers  # <-- pin explicite, par process

    datamodule = get_hl_datamodules(hl_cfg)

    # def _cv_worker(gpu_id, task_queue, result_queue, hl_cfg_dict):

    #     print("_cv_worker gpu_id:", gpu_id)

    #     # Reconstruire un cfg indépendant pour ce process (copie, pas de partage d'état)
    #     hl_cfg = OmegaConf.create(hl_cfg_dict)
    #     hl_cfg.trainer.devices = [gpu_id]  # pin sur le bon GPU
    #     hl_cfg.trainer.accelerator = "gpu"

    #     datamodule = get_hl_datamodules(hl_cfg)

    while True:
        try:
            fold_idx, cv_train_labels, cv_val_labels = task_queue.get_nowait()
        except Empty:
            break

        print(f"\t === fold_idx: {fold_idx} ===")
        print(f"\t\t --- cv_train_labels: {len(cv_train_labels)} ---")
        print(f"\t\t --- cv_val_labels: {len(cv_val_labels)} ---")

        datamodule.train_data.update_indices(cv_train_labels)
        datamodule.val_data.update_indices(cv_val_labels)

        hl_trainer, hl_model = get_hl_modules(hl_cfg, datamodule)

        StartTime = time.time()
        print(f"\t\t --- fit() ---")
        hl_trainer.fit(model=hl_model, datamodule=datamodule, ckpt_path=None)
        print(f"\t\t --- end fit() ---")
        print(f"\t+++ CV fold {fold_idx} on GPU {gpu_id} : {time.time() - StartTime} +++")

        metric_dict = {**hl_trainer.callback_metrics}
        rse = metric_dict["val/reg_loss"] + metric_dict["val/reg_time_loss"]
        score = -torch.sqrt(rse).cpu().numpy()
        result_queue.put((fold_idx, score))


def get_cv_rmse_hl_multigpu(
    hl_cfg, n_splits=3, n_workers_per_gpu=1, max_parallel=None, dataloader_workers=None
):
    n_gpus = torch.cuda.device_count()
    gpu_slots = [g for g in range(n_gpus) for _ in range(n_workers_per_gpu)]
    if max_parallel is not None:
        gpu_slots = gpu_slots[:max_parallel]

    n_parallel_processes = len(gpu_slots)

    # Budget CPU total réparti entre les process GPU lancés en parallèle
    total_cpus = os.cpu_count() or 1

    if dataloader_workers is None:
        # réserve 1 cpu pour le process principal/overhead, répartit le reste
        dataloader_workers = min(
            max(0, (total_cpus - 1) // max(n_parallel_processes, 1)), hl_cfg["num_workers"]
        )

    print(
        f"CPUs total={total_cpus}, process GPU en parallèle={n_parallel_processes}, "
        f"num_workers par process={dataloader_workers}"
    )

    ctx = mp.get_context("spawn")
    task_queue, result_queue = ctx.Queue(), ctx.Queue()

    # def get_cv_rmse_hl_multigpu(hl_cfg, n_splits=3):
    #     ctx = mp.get_context("spawn")

    # datamodule temporaire juste pour récupérer les labels (dans le process parent)
    datamodule = get_hl_datamodules(hl_cfg)
    saved_train_labels = list(datamodule.train_data.labels)

    # cfg sérialisable (dict pur), picklable proprement vers les enfants
    hl_cfg_dict = OmegaConf.to_container(hl_cfg, resolve=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # task_queue = ctx.Queue()
    # result_queue = ctx.Queue()

    fold_idx = 0
    for cv_train_pos, cv_val_pos in kf.split(saved_train_labels):
        cv_train_labels = [saved_train_labels[i] for i in cv_train_pos]
        cv_val_labels = [saved_train_labels[i] for i in cv_val_pos]
        task_queue.put((fold_idx, cv_train_labels, cv_val_labels))
        fold_idx += 1
    n_folds = fold_idx

    n_gpus = torch.cuda.device_count()
    processes = [
        ctx.Process(
            target=_cv_worker,
            args=(gpu_id, task_queue, result_queue, hl_cfg_dict, dataloader_workers),
        )
        for gpu_id in gpu_slots
    ]

    # processes = [
    #     ctx.Process(target=_cv_worker, args=(g, task_queue, result_queue, hl_cfg_dict))
    #     for g in range(n_gpus)
    # ]
    for p in processes:
        p.start()

    results = {}
    for _ in range(n_folds):
        fold_idx, score = result_queue.get()
        results[fold_idx] = score

    for p in processes:
        p.join()

    return np.mean([results[i] for i in range(n_folds)])
