# Per-step AL loop benchmarks

One AL replication's `while True:` loop (`utils/Main/LearningProcedure.py`) does
the following every iteration, each already delimited by its own `+++ ... +++`
timer in the real code:

| # | Step | Real code | What it costs |
|---|------|-----------|----------------|
| 1 | Training | `predictor_model.fit(...)` after `reset_weights`/`reset_trainer` | Full retrain from scratch every iteration, no warm start. Scales with `train_size / batch_size * epochs`. |
| 2 | FullPoolError | `FullPoolErrorFunction` | Predicts on the candidate pool, `pd.concat([df_Train, df_Candidate])`, computes RMSE/MAE/R2/CC over the whole pool. |
| 3 | FullTestError | `FullTestErrorFunction` | `trainer.test()` + `trainer.predict()` on the held-out test/val split. Fixed size, doesn't grow with iterations. |
| 4 | TrainError | `TrainErrorFunction` | Predicts on the current (growing) training set. |
| 5 | CrossValidation | `get_cv_rmse_hl` (skipped by `--no_cv`) | 3-fold CV: retrains the model 3 more times per call. |
| 6 | Selector | `GreedySamplingSelector.select()` | Predicts on candidates (deduped with step 2's prediction when the candidate set matches) + distance/score computation (X cached incrementally across iterations, Y always recomputed since it depends on this iteration's model) + sequential top-k pick. |
| 7 | Bookkeeping | steps 7-8-7-10 in `LearningProcedure.py` | Moves the selected rows from `df_Candidate` to `df_Train` via `pd.concat`/`.drop()`. |

Each script here benchmarks ONE of these steps in isolation, driving the real
production functions (not reimplementations) against a simulated schedule of
`(train_size, candidate_size)` pairs that mimics how one replication actually
evolves: `df_Train` grows by `k_top` and `df_Candidate` shrinks by `k_top`
every iteration, exactly like `LearningProcedure.py`'s loop. This lets you see
how a single step's cost scales with pool size without paying for a full
end-to-end run, and without one step's cost (e.g. training) drowning out
another's in the timing.

Scripts `01`, `02`, `04`, `05`, `06` build a real trainer/model/datamodule once
via `common.build_context()` (same setup `OneIterationFunction.py` does before
the AL loop starts) and reuse it across the whole schedule - the model is
**not actually trained** in these benchmarks (no meaningful `fit()` beyond
what step 1 itself measures), so predictions are garbage; only timing is
measured. Script `03` sweeps test-set size directly instead of the schedule,
since test size doesn't evolve with iterations. Script `07` is pure pandas and
uses synthetic data instead of a real datamodule, since no model/GPU is
involved.

## Usage

Run from `Code/` (matches how `RunSimulation.py` itself is invoked):

```bash
python test/01_training.py --hl-xp baseline_active_learning_local_small
python test/02_full_pool_error.py --hl-xp baseline_active_learning_local_small
python test/03_full_test_error.py --hl-xp baseline_active_learning_local_small
python test/04_train_error.py --hl-xp baseline_active_learning_local_small
python test/05_cross_validation.py --hl-xp baseline_active_learning_local_small
python test/06_selector.py --hl-xp baseline_active_learning_local_small
python test/07_bookkeeping.py
```

Common flags (scripts `01`, `02`, `04`, `05`, `06`; `07` has its own subset,
`03` sweeps `--sizes` instead):

- `--hl-xp` - experiment config to build the datamodule/model from (default:
  `baseline_active_learning_local_small`, needs `data/small_26000.csv` under
  `henrihost-al/`).
- `--strategy` - selector strategy (`iGS`/`GSx`/`GSy`, default `iGS`).
- `--n-train-0` / `--n-candidate-0` - starting sizes for the schedule.
  Defaults are sized to fit the small local dataset (~25,927 usable rows) -
  raise `--hl-xp` to a bigger dataset (or the real `baseline_active_learning`
  config) to benchmark at production scale.
- `--k-top` - candidates moved to train per simulated iteration.
- `--n-iterations` - cap the schedule length (default: run until the
  candidate pool empties).
- `--seed`

Each script prints a table and writes a CSV to `test/results/` for plotting.

## Notes

- `06_selector.py` runs the SAME schedule twice - once with the dX-distance
  cache enabled (default), once forced onto the original uncached fallback
  path - and reports the speedup per iteration directly, so the caching
  change's benefit is visible without needing a before/after code checkout.
  At small local-test scale (a few thousand candidates/train rows, k_top a
  large fraction of train_size) the cache shows little or no benefit - the
  cache's own bookkeeping overhead is comparable to what it saves when there's
  little to reuse. The win is scale-dependent and shows up once train_size
  grows much larger than k_top (production-scale CandidateProportion/k_top);
  rerun with bigger `--n-train-0`/`--n-candidate-0`/`--hl-xp` to see it.
- **Known issue surfaced by `05_cross_validation.py`**: `get_cv_rmse_hl`
  raises `KeyError: 'val/reg_loss'` under
  `baseline_active_learning_local_small.yaml`'s current
  `max_epochs: 5` / `check_val_every_n_epoch: 49`. Lightning only validates
  when `(epoch+1) % check_val_every_n_epoch == 0`, which never happens within
  5 epochs, so `callback_metrics` never gets populated with `val/*` keys. This
  predates this benchmark folder but was invisible until now because every
  real run so far has used `--no_cv`, which skips this function entirely. Not
  fixed here - see the top-level TODO/discuss with the team before changing
  `check_val_every_n_epoch`, since the right fix (e.g. matching it to
  `max_epochs`) trades off CV cost against how early it becomes measurable.
- `train_size + candidate_size` must not exceed the pool size available under
  `--hl-xp` (e.g. ~25,927 for `baseline_active_learning_local_small`); scripts
  raise a clear error if the schedule asks for more than that.
