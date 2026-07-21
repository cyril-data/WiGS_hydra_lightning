"""Shared infrastructure for the per-step AL benchmarks in this folder.

Each benchmark drives the REAL production functions (LearningProcedure.py's
building blocks) against a sequence of (train_size, candidate_size) pairs that
mimics how one AL replication actually evolves: df_Train grows by k_top and
df_Candidate shrinks by k_top every iteration, until the candidate pool is
empty. This lets each step be profiled in isolation, at controlled sizes,
without paying for a full end-to-end run.

build_context() constructs the same trainer/model/datamodule/df_full/df_test
that OneIterationFunction.py builds before the AL loop starts, once. Each
benchmark script then calls slice_pool() at every schedule step to select
(df_Train, df_Candidate) of the requested sizes and sync the datamodule's
train_data indices to match - exactly what LearningProcedure.py's iteration
loop does, just without running the rest of the loop.
"""

import argparse
import csv as csv_module
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Code/ on sys.path

import numpy as np
import psutil
import torch

from utils.Prediction.LightHydra import (
    full_datamodule_to_pd,
    get_hl_cfg,
    get_hl_datamodules,
    get_hl_modules,
    reset_trainer,
    reset_weights,
    update_scheduler,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def make_synthetic_pool(n_rows, n_cols=84, seed=0):
    """A synthetic pandas DataFrame with the same rough shape as the real
    df_full (43 X + 13 y_reg + 13 y_time_reg + 2 y_cls + 13 y_time_cls = 84
    columns for our small dataset) but random values and a RangeIndex. For
    benchmarks of pure pandas/bookkeeping operations that don't need a real
    trained model or GPU, so they don't pay for build_context()'s setup."""
    rng = np.random.default_rng(seed)
    import pandas as pd

    return pd.DataFrame(rng.normal(size=(n_rows, n_cols)).astype(np.float32))


def iteration_schedule(n_train_0, n_candidate_0, k_top, n_iterations=None):
    """(iteration, train_size, candidate_size) triples matching how
    LearningProcedure.py's loop evolves df_Train/df_Candidate: each step moves
    min(k_top, remaining candidates) rows from candidate to train, stopping
    when the candidate pool is empty (or after n_iterations steps if given)."""
    schedule = []
    train_size, candidate_size = n_train_0, n_candidate_0
    it = 0
    while candidate_size > 0:
        schedule.append((it, train_size, candidate_size))
        if n_iterations is not None and it + 1 >= n_iterations:
            break
        step = min(k_top, candidate_size)
        train_size += step
        candidate_size -= step
        it += 1
    return schedule


def build_context(hl_xp, strategy_name="iGS", seed=0, extra_overrides=None):
    """Builds the trainer/model/datamodule/df_full/df_test pool once. Does NOT
    perform the train/candidate split itself - each benchmark script slices
    its own (df_Train, df_Candidate) of the desired sizes out of df_full via
    slice_pool(), so the same pool can be reused across an entire schedule
    without rebuilding the datamodule each time.

    extra_overrides: optional list of Hydra override strings (e.g.
    ["num_workers=0"]) to tune a single experiment-config value without
    needing a new YAML - passed straight through to get_hl_cfg."""
    np.random.seed(seed)

    config = {
        "add_useful_params": {
            "strategy_name": strategy_name,
            "hl_xp": hl_xp,
            "hl_max_epoch": None,
            "hl_worker": None,
            "subset_rand_candidat": None,
            "curriculum": False,
        }
    }
    hl_cfg = get_hl_cfg(config, extra_overrides=extra_overrides)
    project_root = os.environ.get("PROJECT_ROOT", os.path.join(os.getcwd(), ".."))
    hl_cfg["csv_path"] = os.path.join(f"{project_root}/../henrihost-al/", hl_cfg["csv_path"])

    datamodule = get_hl_datamodules(hl_cfg)
    hl_trainer, hl_model = get_hl_modules(hl_cfg, datamodule)

    df_all, y_size = full_datamodule_to_pd(datamodule)

    current_train_labels = datamodule.train_data.index[datamodule.train_data.indices]
    current_test_labels = datamodule.test_data.index[datamodule.test_data.indices]

    df_full = df_all.loc[current_train_labels, :]
    df_test = df_all.loc[current_test_labels, :]

    hl_trainer, hl_model, datamodule, hl_cfg = update_scheduler(
        hl_trainer, hl_model, datamodule, hl_cfg
    )

    return {
        "hl_trainer": hl_trainer,
        "hl_model": hl_model,
        "hl_data": datamodule,
        "hl_cfg": hl_cfg,
        "y_size": y_size,
        "df_full": df_full,
        "df_test": df_test,
        "add_useful_params": config["add_useful_params"],
    }


def slice_pool(ctx, train_size, candidate_size):
    """Selects (df_Train, df_Candidate) of the requested sizes from df_full and
    syncs datamodule.train_data's active indices to match df_Train - the same
    thing LearningProcedure.py does at the top of every iteration
    (`hl_data.train_data.update_indices(X_train_df.index)`). Candidate-side
    dataloader indices (pred_data) are synced by the functions under test
    themselves (FullPoolErrorFunction/TrainErrorFunction/the selector all call
    `hl_data.pred_data.update_indices(...)` internally), so this only handles
    the train side.

    df_full must have at least train_size + candidate_size rows.
    """
    df_full = ctx["df_full"]
    total_needed = train_size + candidate_size
    if total_needed > len(df_full):
        raise ValueError(
            f"train_size + candidate_size = {total_needed} exceeds pool size "
            f"{len(df_full)} - use a smaller schedule or a larger --hl-xp dataset."
        )
    df_Train = df_full.iloc[:train_size]
    df_Candidate = df_full.iloc[train_size : train_size + candidate_size]
    ctx["hl_data"].train_data.update_indices(df_Train.index)
    return df_Train, df_Candidate


def reset_model(ctx):
    """Mirrors LearningProcedure.py's per-iteration reset before fit()."""
    reset_trainer(ctx["hl_trainer"])
    ctx["hl_model"].apply(reset_weights)


_process = psutil.Process()

# Columns every benchmark's rows get via Measure.as_dict() - append to a
# script's own ["iteration", "train_size", "candidate_size", ...] fieldnames.
MEASURE_FIELDS = ["elapsed_s", "rss_delta_mb", "gpu_peak_mb", "gpu_reserved_mb"]


class Measure:
    """Times a block and captures CPU RSS delta + peak GPU memory allocated
    during the block.

    RSS is a delta (after - before), not a peak - Linux doesn't expose
    per-window peak RSS without a background sampling thread, so this only
    tells you net growth across the block, which is enough to see e.g. the
    ~5x/~2x/~1x reductions from the datamodule fix show up as a shrinking
    delta rather than needing an absolute peak.

    GPU peak IS a true per-block peak: torch.cuda.reset_peak_memory_stats()
    on entry means max_memory_allocated()/max_memory_reserved() on exit
    reflect only this block, not accumulated history since process start.
    Both read 0 when no CUDA device is available (e.g. running 07's pure
    pandas benchmark on a CPU-only machine).
    """

    def __enter__(self):
        self.rss_before_mb = _process.memory_info().rss / (1024**2)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc_info):
        self.elapsed = time.time() - self.t0
        rss_after_mb = _process.memory_info().rss / (1024**2)
        self.rss_delta_mb = rss_after_mb - self.rss_before_mb
        if torch.cuda.is_available():
            self.gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            self.gpu_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
        else:
            self.gpu_peak_mb = 0.0
            self.gpu_reserved_mb = 0.0

    def as_dict(self):
        return {
            "elapsed_s": round(self.elapsed, 4),
            "rss_delta_mb": round(self.rss_delta_mb, 2),
            "gpu_peak_mb": round(self.gpu_peak_mb, 2),
            "gpu_reserved_mb": round(self.gpu_reserved_mb, 2),
        }


def write_csv(filename, rows, fieldnames):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nSaved {len(rows)} rows to {path}")


def make_arg_parser(description, default_n_train_0=10_000, default_n_candidate_0=8_900_000):
    """Common CLI shared by every benchmark script. Defaults now target the
    real production case: --hl-xp baseline_active_learning (the full ~9M-row
    dataset, data/database_oct-24.csv), starting from a ~10k-row train set
    with the rest (~8.99M) as candidates, k_top=2000. That candidate pool
    figure is an estimate (10M raw rows minus dropna/k-fold losses) - if it's
    off, slice_pool() raises a clear "exceeds pool size N" error rather than
    failing silently, so it's safe to correct --n-candidate-0 from that
    message on the machine that actually has the resources to run this.

    Override --hl-xp plus these sizes to benchmark against a smaller dataset
    (e.g. --hl-xp baseline_active_learning_local_small, ~25,927 usable rows)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--hl-xp", default="baseline_active_learning")
    p.add_argument("--strategy", default="iGS")
    p.add_argument("--n-train-0", type=int, default=default_n_train_0)
    p.add_argument("--n-candidate-0", type=int, default=default_n_candidate_0)
    p.add_argument("--k-top", type=int, default=2000)
    p.add_argument("--n-iterations", type=int, default=None, help="cap the schedule length")
    p.add_argument("--seed", type=int, default=0)
    return p


def print_table(rows, fieldnames):
    widths = {name: max(len(name), *(len(f"{row[name]}") for row in rows)) for name in fieldnames}
    header = "  ".join(name.ljust(widths[name]) for name in fieldnames)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(f"{row[name]}".ljust(widths[name]) for name in fieldnames))
