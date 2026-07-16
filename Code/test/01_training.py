"""Benchmark step 1: model retrain-from-scratch (LearningProcedure.py's
`predictor_model.fit(...)` call), across a schedule of growing train-set sizes.

This is expected to be the dominant per-iteration cost (see LearningProcedure.py
lines 110-121: hl_model.apply(reset_weights) + reset_trainer + fit(), every
iteration, no warm start). The lever that matters here is epoch count
(--max-epochs via the experiment config), not data size mechanics - this
benchmark exists to make that scaling visible and quantifiable.

Usage (run from Code/):
    python test/01_training.py --hl-xp baseline_active_learning_local_small
"""

from common import (
    build_context,
    iteration_schedule,
    make_arg_parser,
    Measure,
    MEASURE_FIELDS,
    print_table,
    reset_model,
    slice_pool,
    write_csv,
)

FIELDNAMES = ["iteration", "train_size", "candidate_size"] + MEASURE_FIELDS


def main():
    parser = make_arg_parser("Benchmark: model fit() cost vs train set size")
    args = parser.parse_args()

    ctx = build_context(args.hl_xp, strategy_name=args.strategy, seed=args.seed)
    schedule = iteration_schedule(args.n_train_0, args.n_candidate_0, args.k_top, args.n_iterations)

    rows = []
    for it, train_size, candidate_size in schedule:
        slice_pool(ctx, train_size, candidate_size)
        reset_model(ctx)

        with Measure() as m:
            ctx["hl_trainer"].fit(model=ctx["hl_model"], datamodule=ctx["hl_data"], ckpt_path=None)

        row = {"iteration": it, "train_size": train_size, "candidate_size": candidate_size, **m.as_dict()}
        rows.append(row)
        print(
            f"iter {it}: train={train_size} candidate={candidate_size} -> {m.elapsed:.3f}s, "
            f"RSS delta {m.rss_delta_mb:+.1f}MB, GPU peak {m.gpu_peak_mb:.1f}MB "
            f"(reserved {m.gpu_reserved_mb:.1f}MB)"
        )

    print()
    print_table(rows, FIELDNAMES)
    write_csv("01_training.csv", rows, FIELDNAMES)


if __name__ == "__main__":
    main()
