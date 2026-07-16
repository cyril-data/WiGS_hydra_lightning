"""Benchmark step 3: FullTestErrorFunction - runs trainer.test() + predict() on
the held-out test/val split (LearningProcedure.py lines 146-155).

Unlike the other steps, the test set does NOT grow/shrink across AL
iterations - it's fixed for the whole replication. So instead of driving this
against the candidate/train schedule, this benchmark sweeps the TEST SUBSET
SIZE directly, to answer "does this step's cost even depend on size, and by
how much" - useful context for interpreting why it shows up roughly flat in
the real per-iteration logs.

The model here is NOT trained - predictions are meaningless, only
timing/memory is measured.

Usage (run from Code/):
    python test/03_full_test_error.py --hl-xp baseline_active_learning_local_small
"""

import argparse

from common import build_context, Measure, MEASURE_FIELDS, print_table, write_csv

from utils.Prediction.FullPoolError import FullTestErrorFunction

FIELDNAMES = ["test_size"] + MEASURE_FIELDS


def main():
    parser = argparse.ArgumentParser(description="Benchmark: FullTestErrorFunction cost vs test size")
    parser.add_argument("--hl-xp", default="baseline_active_learning_local_small")
    parser.add_argument("--strategy", default="iGS")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=None, help="test subset sizes to sweep"
    )
    args = parser.parse_args()

    ctx = build_context(args.hl_xp, strategy_name=args.strategy, seed=args.seed)
    df_test_full = ctx["df_test"]

    sizes = args.sizes
    if sizes is None:
        n = len(df_test_full)
        sizes = sorted(set(min(n, s) for s in (100, 500, 1000, 2000, n)))

    rows = []
    for size in sizes:
        df_test = df_test_full.iloc[:size]

        with Measure() as m:
            FullTestErrorFunction(
                InputModel=ctx["hl_trainer"],
                SimulationConfigInputUpdated=ctx,
                df_test=df_test,
                y_size=ctx["y_size"],
            )

        row = {"test_size": size, **m.as_dict()}
        rows.append(row)
        print(
            f"test_size={size} -> {m.elapsed:.3f}s, RSS delta {m.rss_delta_mb:+.1f}MB, "
            f"GPU peak {m.gpu_peak_mb:.1f}MB (reserved {m.gpu_reserved_mb:.1f}MB)"
        )

    print()
    print_table(rows, FIELDNAMES)
    write_csv("03_full_test_error.csv", rows, FIELDNAMES)


if __name__ == "__main__":
    main()
