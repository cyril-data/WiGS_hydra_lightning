"""Benchmark step 4: TrainErrorFunction - predicts on the current training set
and computes RMSE/MAE/R2/CC against it (LearningProcedure.py lines 157-166).

Cost driver: a predict() pass over the whole (growing) df_Train every
iteration.

The model here is NOT trained - predictions are meaningless, only
timing/memory is measured.

Usage (run from Code/):
    python test/04_train_error.py --hl-xp baseline_active_learning_local_small
"""

from common import (
    build_context,
    iteration_schedule,
    make_arg_parser,
    Measure,
    MEASURE_FIELDS,
    print_table,
    slice_pool,
    write_csv,
)

from utils.Prediction.FullPoolError import TrainErrorFunction

FIELDNAMES = ["iteration", "train_size", "candidate_size"] + MEASURE_FIELDS


def main():
    parser = make_arg_parser("Benchmark: TrainErrorFunction cost vs train_size")
    args = parser.parse_args()

    ctx = build_context(args.hl_xp, strategy_name=args.strategy, seed=args.seed)
    schedule = iteration_schedule(args.n_train_0, args.n_candidate_0, args.k_top, args.n_iterations)

    rows = []
    for it, train_size, candidate_size in schedule:
        df_Train, _ = slice_pool(ctx, train_size, candidate_size)

        with Measure() as m:
            TrainErrorFunction(
                InputModel=ctx["hl_trainer"],
                SimulationConfigInputUpdated=ctx,
                df_train=df_Train,
                y_size=ctx["y_size"],
            )

        row = {"iteration": it, "train_size": train_size, "candidate_size": candidate_size, **m.as_dict()}
        rows.append(row)
        print(
            f"iter {it}: train={train_size} candidate={candidate_size} -> {m.elapsed:.3f}s, "
            f"RSS delta {m.rss_delta_mb:+.1f}MB, GPU peak {m.gpu_peak_mb:.1f}MB "
            f"(reserved {m.gpu_reserved_mb:.1f}MB)"
        )

    print()
    print_table(rows, FIELDNAMES)
    write_csv("04_train_error.csv", rows, FIELDNAMES)


if __name__ == "__main__":
    main()
