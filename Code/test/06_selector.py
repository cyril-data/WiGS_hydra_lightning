"""Benchmark step 6: GreedySamplingSelector.select() - candidate prediction +
distance/score computation + sequential top-k pick (LearningProcedure.py lines
210-219).

Runs the SAME schedule through two selector instances to quantify the
dX-distance-caching optimization directly: one with caching enabled (the
default), one with caching forced off (via `_dx_cache_max_bytes = 0`, which
makes every call fall back to the original chunked, non-materializing
min_igs_per_row_gpu path). X is cached/extended incrementally across
iterations since it's iteration-invariant; Y is always recomputed in full
since it depends on this iteration's freshly retrained model's predictions on
the candidates - see GreedySamplingSelector.py's _get_dx_matrix docstring.

The cache trades GPU memory (the growing dX matrix, held across iterations)
for compute - this benchmark reports GPU peak for both paths side by side so
that tradeoff is visible, not just the speedup.

The model gets ONE warm-up fit() before the schedule runs - required because
the selector's hydralightning path reads Trainer.model, which Lightning only
populates after fit() has run at least once. Weights are otherwise never
updated again during the benchmark; predictions are not meaningful, only
timing/memory is measured.

Usage (run from Code/):
    python test/06_selector.py --hl-xp baseline_active_learning_local_small
"""

import torch

from common import (
    build_context,
    iteration_schedule,
    make_arg_parser,
    Measure,
    print_table,
    slice_pool,
    write_csv,
)

from utils.Selector.GreedySamplingSelector import GreedySamplingSelector

SUMMARY_FIELDNAMES = [
    "iteration",
    "train_size",
    "candidate_size",
    "cached_s",
    "uncached_s",
    "speedup_x",
    "cached_gpu_peak_mb",
    "uncached_gpu_peak_mb",
    "cached_rss_delta_mb",
    "uncached_rss_delta_mb",
]


def run_schedule(ctx, schedule, selector):
    # SimulationConfigInputUpdated["df_Candidate"] is X-only in the real pipeline
    # (built from df_full.iloc[:, :x_size] in OneIterationFunction.py - only
    # df_Train gets re-fetched with full X+Y columns afterward) and that's what
    # actually reaches selector_model.select(). Passing the full X+Y slice here
    # instead breaks get_features_and_target(df_Candidate, y_size=None)'s implicit
    # "no split, return everything as X" contract and mismatches X_Train's width.
    x_size = ctx["df_full"].shape[1] - ctx["y_size"]

    rows = []
    for it, train_size, candidate_size in schedule:
        df_Train, df_Candidate = slice_pool(ctx, train_size, candidate_size)
        df_Candidate = df_Candidate.iloc[:, :x_size]

        with Measure() as m:
            selector.select(
                df_Candidate=df_Candidate,
                df_Train=df_Train,
                y_size=ctx["y_size"],
                Model=ctx["hl_trainer"],
                SimulationConfigInputUpdated=ctx,
            )

        rows.append(
            {"iteration": it, "train_size": train_size, "candidate_size": candidate_size, **m.as_dict()}
        )
    return rows


def main():
    parser = make_arg_parser("Benchmark: GreedySamplingSelector.select() cost, cached vs uncached")
    parser.add_argument(
        "--batch-size", type=int, default=8192, help="candidate-axis GPU chunk size (was hardcoded to 512)"
    )
    parser.add_argument(
        "--train-chunk-size", type=int, default=None, help="train-axis GPU chunk size (defaults to --batch-size)"
    )
    parser.add_argument(
        "--fp32", action="store_true", help="disable fp16 for the distance matmuls (default: fp16 on)"
    )
    parser.add_argument(
        "--profile-xy", action="store_true",
        help="print X-distance vs Y-distance wall-clock time per iteration (iGS only; adds "
             "cuda.synchronize() calls, so this run will be slower than without it)",
    )
    parser.add_argument(
        "--override", nargs="+", default=None,
        help="extra Hydra overrides for the experiment config, e.g. --override num_workers=0 "
             "(tune a single value without a new YAML)",
    )
    args = parser.parse_args()
    dtype = torch.float32 if args.fp32 else torch.float16

    ctx = build_context(
        args.hl_xp, strategy_name=args.strategy, seed=args.seed, extra_overrides=args.override
    )
    schedule = iteration_schedule(args.n_train_0, args.n_candidate_0, args.k_top, args.n_iterations)

    # The selector's hydralightning path does Model.predict(model=Model.model, ...)
    # - Trainer.model is only populated after fit() has run at least once. In the
    # real loop training always precedes the selector call every iteration; here
    # we only need it populated once since we're not re-benchmarking training.
    slice_pool(ctx, schedule[0][1], schedule[0][2])
    ctx["hl_trainer"].fit(model=ctx["hl_model"], datamodule=ctx["hl_data"], ckpt_path=None)

    cached_selector = GreedySamplingSelector(
        strategy=args.strategy,
        k_top_candidate=args.k_top,
        batch_size=args.batch_size,
        train_chunk_size=args.train_chunk_size,
        dtype=dtype,
        profile_xy=args.profile_xy,
    )
    print("--- cached (default) ---")
    cached_rows = run_schedule(ctx, schedule, cached_selector)
    for r in cached_rows:
        print(f"iter {r['iteration']}: train={r['train_size']} candidate={r['candidate_size']} "
              f"-> {r['elapsed_s']:.3f}s, GPU peak {r['gpu_peak_mb']:.1f}MB")

    uncached_selector = GreedySamplingSelector(
        strategy=args.strategy,
        k_top_candidate=args.k_top,
        batch_size=args.batch_size,
        train_chunk_size=args.train_chunk_size,
        dtype=dtype,
        profile_xy=args.profile_xy,
    )
    uncached_selector._dx_cache_max_bytes = 0  # force every call through the fallback path
    print("\n--- uncached (fallback path, same as before the dX-caching change) ---")
    uncached_rows = run_schedule(ctx, schedule, uncached_selector)
    for r in uncached_rows:
        print(f"iter {r['iteration']}: train={r['train_size']} candidate={r['candidate_size']} "
              f"-> {r['elapsed_s']:.3f}s, GPU peak {r['gpu_peak_mb']:.1f}MB")

    rows = []
    for cached, uncached in zip(cached_rows, uncached_rows):
        speedup = (uncached["elapsed_s"] / cached["elapsed_s"]) if cached["elapsed_s"] > 0 else float("nan")
        rows.append(
            {
                "iteration": cached["iteration"],
                "train_size": cached["train_size"],
                "candidate_size": cached["candidate_size"],
                "cached_s": cached["elapsed_s"],
                "uncached_s": uncached["elapsed_s"],
                "speedup_x": round(speedup, 2),
                "cached_gpu_peak_mb": cached["gpu_peak_mb"],
                "uncached_gpu_peak_mb": uncached["gpu_peak_mb"],
                "cached_rss_delta_mb": cached["rss_delta_mb"],
                "uncached_rss_delta_mb": uncached["rss_delta_mb"],
            }
        )

    print()
    print_table(rows, SUMMARY_FIELDNAMES)
    write_csv("06_selector.csv", rows, SUMMARY_FIELDNAMES)


if __name__ == "__main__":
    main()
