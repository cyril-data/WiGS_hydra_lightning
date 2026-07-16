"""Benchmark step 2: FullPoolErrorFunction - predicts on the candidate pool,
builds the "hybrid" (train truth + candidate prediction) vector, and computes
RMSE/MAE/R2/CC over the whole pool (LearningProcedure.py lines 128-144).

Cost drivers: a predict() pass over df_Candidate (now shared/cached with the
selector's own prediction via the dedup fix, but this benchmark calls
FullPoolErrorFunction alone so it still pays that cost - see
06_selector.py for the dedup'd combination) plus a pd.concat([df_Train,
df_Candidate]) over the whole remaining pool every call.

The model here is NOT trained (no fit() is called) - predictions are
meaningless, only the timing/memory characteristics of the
predict/concat/metric pipeline are being measured.

Usage (run from Code/):
    python test/02_full_pool_error.py --hl-xp baseline_active_learning_local_small
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

from utils.Prediction.FullPoolError import FullPoolErrorFunction

FIELDNAMES = ["iteration", "train_size", "candidate_size"] + MEASURE_FIELDS


def main():
    parser = make_arg_parser("Benchmark: FullPoolErrorFunction cost vs (train_size, candidate_size)")
    args = parser.parse_args()

    ctx = build_context(args.hl_xp, strategy_name=args.strategy, seed=args.seed)
    schedule = iteration_schedule(args.n_train_0, args.n_candidate_0, args.k_top, args.n_iterations)

    rows = []
    for it, train_size, candidate_size in schedule:
        df_Train, df_Candidate = slice_pool(ctx, train_size, candidate_size)
        ctx["df_Train"] = df_Train

        with Measure() as m:
            FullPoolErrorFunction(
                InputModel=ctx["hl_trainer"],
                SimulationConfigInputUpdated=ctx,
                df_Candidate=df_Candidate,
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
    write_csv("02_full_pool_error.csv", rows, FIELDNAMES)


if __name__ == "__main__":
    main()
