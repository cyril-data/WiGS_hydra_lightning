"""Benchmark step 7: the post-selection index bookkeeping
(LearningProcedure.py lines 221-252, its own "#7-8-7-10" timer): moving the
selected rows from df_Candidate to df_Train via pd.concat/.drop().

This step is pure pandas (no model/GPU involved - GPU columns in the output
will read 0), so unlike the other benchmarks it uses a synthetic DataFrame
instead of build_context() - no need to pay for Lightning/datamodule setup
just to time a concat/drop. RSS delta is the metric that matters here: each
pd.concat/.drop() call allocates a new DataFrame, and this makes that
per-iteration memory churn visible.

Faithfully replicates the real code's redundant double-fetch of
QueryObservation (once from df_Candidate, immediately overwritten by a second
fetch from df_full) - see LearningProcedure.py lines 226 and 238 - since the
point here is to measure the CURRENT cost of this block, not an idealized one.

Usage (run from Code/):
    python test/07_bookkeeping.py
"""

import argparse

import pandas as pd

from common import iteration_schedule, make_synthetic_pool, Measure, MEASURE_FIELDS, print_table, write_csv

FIELDNAMES = ["iteration", "train_size", "candidate_size"] + MEASURE_FIELDS


def main():
    parser = argparse.ArgumentParser(description="Benchmark: post-selection pd.concat/.drop() cost")
    parser.add_argument("--n-train-0", type=int, default=5000)
    parser.add_argument("--n-candidate-0", type=int, default=15000)
    parser.add_argument("--k-top", type=int, default=2000)
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--n-cols", type=int, default=84, help="matches real df_full's column count")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    schedule = iteration_schedule(args.n_train_0, args.n_candidate_0, args.k_top, args.n_iterations)
    n_total = args.n_train_0 + args.n_candidate_0
    df_full = make_synthetic_pool(n_total, n_cols=args.n_cols, seed=args.seed)

    rows = []
    for it, train_size, candidate_size in schedule:
        df_Train = df_full.iloc[:train_size]
        df_Candidate = df_full.iloc[train_size : train_size + candidate_size]
        step = min(args.k_top, candidate_size)
        QueryObservationIndex = df_Candidate.index[:step]

        with Measure() as m:
            # Faithful to LearningProcedure.py: the df_Candidate-sourced fetch is
            # immediately discarded and refetched from df_full - real behavior,
            # not something this benchmark should "fix".
            QueryObservation = df_Candidate.loc[QueryObservationIndex]
            QueryObservation = df_full.loc[QueryObservationIndex, :]
            df_Train = pd.concat([df_Train, QueryObservation])
            df_Candidate = df_Candidate.drop(QueryObservationIndex)

        row = {"iteration": it, "train_size": train_size, "candidate_size": candidate_size, **m.as_dict()}
        rows.append(row)
        print(
            f"iter {it}: train={train_size} candidate={candidate_size} -> {m.elapsed:.4f}s, "
            f"RSS delta {m.rss_delta_mb:+.2f}MB"
        )

    print()
    print_table(rows, FIELDNAMES)
    write_csv("07_bookkeeping.csv", rows, FIELDNAMES)


if __name__ == "__main__":
    main()
