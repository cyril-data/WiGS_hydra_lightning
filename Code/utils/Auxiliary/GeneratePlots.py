### Import Packages ###
import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt
import ast
import yaml
import traceback


def safe_literal_eval(x):
    if pd.isna(x):
        return []
    return ast.literal_eval(x)


def flag_to_int(value):
    """Coerce a SimulationParameters.yaml flag to 0/1 - values are stringified
    Python bools (e.g. "True"/"False") since SimulationParameters is built via
    str(...) upstream, but handle real bools too in case that ever changes."""
    if isinstance(value, bool):
        return int(value)
    return int(str(value).strip().lower() == "true")


def IndividualTracesPlot(
    Subtitle=None,
    TransparencyVal=0.85,
    CriticalValue=None,  # accepted for call-site compatibility with MeanVariancePlot; unused here (no CI band on raw traces)
    RelativeError=None,
    Colors=None,
    Linestyles=None,
    RandomLabels=None,
    xlim=None,
    Y_Label=None,
    VarInput=None,  # accepted for call-site compatibility with MeanVariancePlot; unused here (no variance computed)
    initial_train_size: int = None,
    k_top: int = 1,
    FigSize=(9, 12),
    LegendMapping=None,
    show_legend=True,
    total_pool_size=None,
    sim_name=None,  # dict: {Label: {true_Sim_id: descriptive_name}}. If given, used to
    # build the legend label as f"{Label} - {sim_name[Label][true_id]}".
    # `true_id` is looked up via `sim_ids` if provided, otherwise via the
    # positional Sim_id (see `sim_ids` below for why this matters).
    sim_ids=None,  # dict: {Label: [true_Sim_id, ...]}, one true id per column of
    # Results / k_top[Label], IN THE SAME ORDER as those columns.
    # Needed because "Sim_0", "Sim_1"... in the DataFrame are just
    # positional column names -- they do NOT necessarily match the
    # real simulation id (param["Sim"]) used as a key in `sim_name`,
    # e.g. build it as: sim_ids[strategy] = [p["Sim"] for p in params[strategy]]
    TraceColors=None,  # dict: {(Label, true_Sim_id): color}. Pins an exact color to one
    # specific trace. Takes priority over everything else.
    colormap="turbo",  # colormap used to auto-generate one distinct color per trace,
    # so every individual sim can be told apart even within the
    # same Label. Only applies to traces not covered by
    # TraceColors or Colors (see below).
    linewidth=1.3,
    grid_points: int = 200,
    **SimulationErrorResults,
):
    """
    Plots every individual simulation trace (no mean, no variance, no
    confidence band) on a single graph. One line per (Label, Sim) pair.

    Color rule (every individual trace must be distinguishable):
      - `TraceColors[(Label, true_Sim_id)]`, if given, wins -- pins one exact color.
      - Else, `Colors[Label]`, if given, is used for every sim of that Label
        (legacy behaviour: sims of that Label won't be distinguishable by color,
        only by legend text -- use this only when you deliberately want one
        color per strategy instead of per sim).
      - Else, a color is auto-assigned from `colormap`, spread evenly across
        ALL traces that fall into this case, so every one of them is visually
        distinct regardless of which Label it belongs to.

    Linestyle rule (this is what marks Random vs non-Random, independently of color):
      - Any Label matching "Random" (case-insensitive, or listed explicitly
        in `RandomLabels`) is drawn DASHED.
      - Every other Label is drawn SOLID.
      - `Linestyles` (dict: Label -> linestyle), if provided, always wins
        over this default rule for that Label.

    If RelativeError=<baseline Label> is given, each individual trace is
    plotted as (Method sim - Baseline mean), like MeanVariancePlot does for
    the averaged curve, but here you see the per-sim dispersion around 0
    instead of a single averaged line. This requires interpolating every
    sim onto a shared percent-of-pool grid (see `grid_points`), since a
    sim's own x-axis and the baseline's mean x-axis don't line up exactly.

    Handles the same two k_top cases as MeanVariancePlot:
      - k_top as a dict of per-label DataFrames (columns Sim_0, Sim_1, ...,
        values = lists of selected indices per iteration; a sim that has
        stopped shows an empty list `[]`), i.e. variable / per-sim k_top.
      - k_top as a scalar int, i.e. fixed selection size every iteration
        for every sim.
    """
    if initial_train_size is None:
        raise ValueError("IndividualTracesPlot requires 'initial_train_size' to be provided.")

    RandomLabels = set(RandomLabels) if RandomLabels else set()

    def _is_random(label):
        return label in RandomLabels or "random" in label.lower()

    def _linestyle_for(label):
        if Linestyles and label in Linestyles:
            return Linestyles[label]
        return "--" if _is_random(label) else "-"

    def _per_sim_x_pct(Label, Results, n_simulations):
        """Returns list of (sim_idx, sim_id, x_pct, y_sim) tuples, one per sim, real (non-interpolated) axis.
        sim_idx = positional column index (0, 1, 2...); sim_id = column name (e.g. "Sim_0"),
        which is only a positional label and may NOT match the true simulation identity."""
        pairs = []
        if isinstance(k_top, dict) and Label in k_top:
            sim_cols = [c for c in k_top[Label].columns if c.startswith("Sim_")]
            selection_sizes = k_top[Label][sim_cols].map(len)

            per_sim_x_abs = {}
            per_sim_valid_len = {}
            max_reach = 0
            for col in sim_cols:
                sizes = selection_sizes[col].values
                finished_mask = sizes == 0
                last_active = int(np.argmax(finished_mask)) if finished_mask.any() else len(sizes)
                cum = initial_train_size + np.cumsum(sizes[:last_active])
                x_abs = np.concatenate(([initial_train_size], cum))
                per_sim_x_abs[col] = x_abs
                per_sim_valid_len[col] = last_active + 1
                max_reach = max(max_reach, x_abs[-1])

            # pool_size = total_pool_size if total_pool_size is not None else max_reach

            if isinstance(total_pool_size, dict):
                pool_size = total_pool_size.get(Label, max_reach)
            else:
                pool_size = total_pool_size if total_pool_size is not None else max_reach

            for sim_idx, col in enumerate(sim_cols):
                if sim_idx >= n_simulations:
                    break
                x_abs = per_sim_x_abs[col]
                n_pts = per_sim_valid_len[col]
                x_pct = x_abs / pool_size * 100
                y_sim = Results[:n_pts, sim_idx]
                if len(y_sim) != len(x_pct):
                    n_common = min(len(y_sim), len(x_pct))
                    y_sim = y_sim[:n_common]
                    x_pct = x_pct[:n_common]
                if len(x_pct) >= 2:
                    pairs.append((sim_idx, col, x_pct, y_sim))
        else:
            k_top_scalar = k_top if isinstance(k_top, (int, float)) else 1
            num_iterations = Results.shape[0]
            default_pool = initial_train_size + k_top_scalar * num_iterations
            if isinstance(total_pool_size, dict):
                pool_size = total_pool_size.get(Label, default_pool)
            else:
                pool_size = total_pool_size if total_pool_size is not None else default_pool
            iterations_array = np.arange(num_iterations)
            x_abs = initial_train_size + k_top_scalar * iterations_array
            x_pct = x_abs / pool_size * 100
            for sim_idx in range(n_simulations):
                pairs.append((sim_idx, f"Sim_{sim_idx}", x_pct, Results[:, sim_idx]))
        return pairs

    # ------------------------------------------------------------------
    # Pre-extract (x_pct, y) per sim per label from raw inputs.
    # ------------------------------------------------------------------
    RawTraces = {}  # Label -> list of (x_pct, y) per sim
    for Label, Results in SimulationErrorResults.items():
        Results = np.asarray(Results)
        n_simulations = Results.shape[1]
        RawTraces[Label] = _per_sim_x_pct(Label, Results, n_simulations)

    # ------------------------------------------------------------------
    # If RelativeError requested: compute baseline mean on a shared grid,
    # then subtract it (interpolated) from every individual sim of every
    # label, including the baseline's own sims (which will hover near 0).
    # ------------------------------------------------------------------
    BaselineInterp = None
    if RelativeError:
        if RelativeError not in RawTraces:
            print(f"  > Warning: Baseline '{RelativeError}' not found. Skipping normalization.")
        else:
            common_grid_pct = np.linspace(0.0, 100.0, num=grid_points)
            baseline_pairs = RawTraces[RelativeError]
            baseline_matrix = np.full((len(common_grid_pct), len(baseline_pairs)), np.nan)
            for sim_idx, (_, sim_id, x_pct, y_sim) in enumerate(baseline_pairs):
                valid_mask = (common_grid_pct >= x_pct[0]) & (common_grid_pct <= x_pct[-1])
                baseline_matrix[valid_mask, sim_idx] = np.interp(
                    common_grid_pct[valid_mask], x_pct, y_sim
                )
            baseline_mean = np.nanmean(baseline_matrix, axis=1)
            BaselineInterp = (common_grid_pct, baseline_mean)
            Y_Label = f"Error Difference (Method - {RelativeError})"

    fig, ax = plt.subplots(figsize=FigSize)
    _warned_labels = set()  # avoid spamming one warning per label

    # ------------------------------------------------------------------
    # Pass 1: resolve every trace's true_sim_id + legend label + y-values,
    # without assigning colors yet (need the total count first).
    # ------------------------------------------------------------------
    ResolvedTraces = []  # list of dicts: Label, true_sim_id, x_pct, y_plot, trace_label
    for Label, pairs in RawTraces.items():
        legend_label = LegendMapping.get(Label, Label) if LegendMapping else Label

        for sim_idx, sim_id, x_pct, y_sim in pairs:
            if BaselineInterp is not None:
                common_grid_pct, baseline_mean = BaselineInterp
                baseline_at_x = np.interp(x_pct, common_grid_pct, baseline_mean)
                y_plot = y_sim - baseline_at_x
            else:
                y_plot = y_sim

            # Resolve the TRUE simulation id for this column: prefer the
            # explicit positional mapping (sim_ids[Label][sim_idx]), since
            # "Sim_0", "Sim_1"... in the DataFrame are just positional column
            # names and do not necessarily match the real param["Sim"].
            true_sim_id = sim_id
            if sim_ids and Label in sim_ids:
                if sim_idx < len(sim_ids[Label]):
                    true_sim_id = sim_ids[Label][sim_idx]
                elif Label not in _warned_labels:
                    print(
                        f"  > Warning: sim_ids['{Label}'] has fewer entries than "
                        f"columns in Results; falling back to positional Sim_id."
                    )
                    _warned_labels.add(Label)

            descriptive_name = None
            if sim_name and Label in sim_name:
                descriptive_name = sim_name[Label].get(true_sim_id)
                if descriptive_name is None and Label not in _warned_labels:
                    print(
                        f"  > Warning: sim_name['{Label}'] has no entry for "
                        f"'{true_sim_id}' (available: {list(sim_name[Label].keys())}); "
                        f"falling back to raw Sim_id in legend."
                    )
                    _warned_labels.add(Label)

            trace_label = (
                f"{legend_label} - {descriptive_name}"
                if descriptive_name is not None
                else f"{legend_label} ({true_sim_id})"
            )

            ResolvedTraces.append(
                {
                    "Label": Label,
                    "true_sim_id": true_sim_id,
                    "x_pct": x_pct,
                    "y_plot": y_plot,
                    "trace_label": trace_label,
                }
            )

    # ------------------------------------------------------------------
    # Pass 2: assign a color to every trace.
    #   TraceColors[(Label, true_sim_id)] > Colors[Label] > auto colormap.
    # Auto colors are spread evenly across ALL traces needing one, so every
    # sim is visually distinguishable regardless of its Label.
    # ------------------------------------------------------------------
    auto_needed_idx = [
        i
        for i, t in enumerate(ResolvedTraces)
        if not (TraceColors and (t["Label"], t["true_sim_id"]) in TraceColors)
        and not (Colors and t["Label"] in Colors)
    ]
    cmap = plt.get_cmap(colormap)
    n_auto = len(auto_needed_idx)
    auto_palette = [cmap(p) for p in np.linspace(0.05, 0.95, n_auto)] if n_auto > 0 else []
    auto_color_for_idx = dict(zip(auto_needed_idx, auto_palette))

    # ------------------------------------------------------------------
    # Pass 3: plot.
    # ------------------------------------------------------------------
    for i, t in enumerate(ResolvedTraces):
        Label, true_sim_id = t["Label"], t["true_sim_id"]

        if TraceColors and (Label, true_sim_id) in TraceColors:
            color = TraceColors[(Label, true_sim_id)]
        elif Colors and Label in Colors:
            color = Colors[Label]
        else:
            color = auto_color_for_idx[i]

        linestyle = _linestyle_for(Label)

        ax.plot(
            t["x_pct"],
            t["y_plot"],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=TransparencyVal,
            label=t["trace_label"],
        )

    ax.set_xlabel("Percent of Learning Pool Labeled")
    ax.set_ylabel(Y_Label)

    if RelativeError and BaselineInterp is not None:
        ax.axhline(y=0.0, color="r", linestyle="-", linewidth=1, alpha=0.5)

    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    if isinstance(xlim, list):
        ax.set_xlim(xlim)

    return fig


### Plotting Function ###
def MeanVariancePlot(
    Subtitle=None,
    TransparencyVal=0.2,
    CriticalValue=1.96,
    RelativeError=None,
    Colors=None,
    Linestyles=None,
    xlim=None,
    Y_Label=None,
    VarInput=False,
    initial_train_size: int = None,
    k_top: int = 1,
    FigSize=(9, 12),
    LegendMapping=None,
    show_legend=True,
    total_pool_size=None,
    sim_name=None,
    grid_points: int = 200,
    **SimulationErrorResults,
):
    """
    Generates trace plots.
    If RelativeError is provided, plots the DIFFERENCE (Method - Baseline).

    Handles k_top varying per simulation AND per iteration (k_top passed as a
    dict of per-label DataFrames of selected-index lists), including
    simulations that stop early (empty list once their pool is exhausted).
    Every simulation gets its own cumulative x-axis, and all
    simulations/labels are interpolated onto ONE shared percent-of-pool grid
    before being averaged or compared -- this is what makes the
    `RelativeError` subtraction (Method - Baseline) safe even when Method and
    Baseline had different selection rhythms.
    """
    if initial_train_size is None:
        raise ValueError("MeanVariancePlot requires 'initial_train_size' to be provided.")

    MeanVector, VarianceVector, StdErrorVector, StdErrorVarianceVector = {}, {}, {}, {}

    # ------------------------------------------------------------------
    # 0. One shared grid (in % of pool labeled) for every label, so that
    #    RelativeError subtraction and cross-label comparison are always
    #    well-defined (no more per-label x-axis mismatches).
    # ------------------------------------------------------------------
    common_grid_pct = np.linspace(0.0, 100.0, num=grid_points)

    InterpolatedMatrices = {}  # Label -> (grid_points x n_sims) matrix
    ValidCounts = {}  # Label -> nb of sims with data at each grid point

    ### Extract ###
    for Label, Results in SimulationErrorResults.items():

        # Results may arrive as a pandas DataFrame (columns = Sim_0, Sim_1, ...).
        # Force it to a plain numpy array so positional (slice, int) indexing
        # below always works, regardless of what was passed in.
        Results = np.asarray(Results)
        n_simulations = Results.shape[1]

        if isinstance(k_top, dict) and Label in k_top:
            # --- k_top varies per sim / per iteration ---
            sim_cols = [c for c in k_top[Label].columns if c.startswith("Sim_")]
            selection_sizes = k_top[Label][sim_cols].map(len)

            interpolated = np.full((len(common_grid_pct), n_simulations), np.nan)

            per_sim_x_abs = {}
            per_sim_valid_len = {}
            max_reach = 0
            for col in sim_cols:
                sizes = selection_sizes[col].values
                finished_mask = sizes == 0

                if finished_mask.any():
                    last_active = int(np.argmax(finished_mask))  # first index sim stops
                else:
                    last_active = len(sizes)

                cum = initial_train_size + np.cumsum(sizes[:last_active])
                x_abs = np.concatenate(([initial_train_size], cum))
                per_sim_x_abs[col] = x_abs
                per_sim_valid_len[col] = last_active + 1  # +1 for initial point
                max_reach = max(max_reach, x_abs[-1])

            if isinstance(total_pool_size, dict):
                pool_size = total_pool_size.get(Label, max_reach)
            else:
                pool_size = total_pool_size if total_pool_size is not None else max_reach

            for sim_idx, col in enumerate(sim_cols):
                if sim_idx >= n_simulations:
                    break
                x_abs = per_sim_x_abs[col]
                n_pts = per_sim_valid_len[col]
                x_pct = x_abs / pool_size * 100

                y_sim = Results[:n_pts, sim_idx]  # this sim's own errors, truncated
                if len(y_sim) != len(x_pct):
                    n_common = min(len(y_sim), len(x_pct))
                    y_sim = y_sim[:n_common]
                    x_pct = x_pct[:n_common]

                if len(x_pct) < 2:
                    continue  # not enough points to interpolate this sim

                # never extrapolate beyond what this sim actually reached
                valid_mask = (common_grid_pct >= x_pct[0]) & (common_grid_pct <= x_pct[-1])
                interpolated[valid_mask, sim_idx] = np.interp(
                    common_grid_pct[valid_mask], x_pct, y_sim
                )

        else:
            # --- k_top scalaire (int) : meme axe x deterministe pour chaque sim ---
            num_iterations = Results.shape[0]
            iterations_array = np.arange(num_iterations)
            default_pool = initial_train_size + k_top * num_iterations
            if isinstance(total_pool_size, dict):
                pool_size = total_pool_size.get(Label, default_pool)
            else:
                pool_size = total_pool_size if total_pool_size is not None else default_pool
            x_abs = initial_train_size + k_top * iterations_array
            x_pct = x_abs / pool_size * 100

            interpolated = np.full((len(common_grid_pct), n_simulations), np.nan)
            valid_mask = (common_grid_pct >= x_pct[0]) & (common_grid_pct <= x_pct[-1])
            for sim_idx in range(n_simulations):
                interpolated[valid_mask, sim_idx] = np.interp(
                    common_grid_pct[valid_mask], x_pct, Results[:, sim_idx]
                )

        InterpolatedMatrices[Label] = interpolated
        ValidCounts[Label] = np.sum(~np.isnan(interpolated), axis=1)

    # ------------------------------------------------------------------
    # 1. Derive mean / variance / std-error from the interpolated matrices.
    #    nanmean/nanvar ignore sims already finished at a given grid point,
    #    instead of silently treating them as 0.
    # ------------------------------------------------------------------
    for Label, interpolated in InterpolatedMatrices.items():
        valid_counts = ValidCounts[Label]
        safe_counts = np.maximum(valid_counts, 1)

        MeanVector[Label] = np.nanmean(interpolated, axis=1)
        VarianceVector[Label] = np.nanvar(interpolated, axis=1)
        StdErrorVector[Label] = np.nanstd(interpolated, axis=1) / np.sqrt(safe_counts)

        df = np.maximum(valid_counts - 1, 1)
        lower_chi2 = chi2.ppf(0.025, df=df)
        upper_chi2 = chi2.ppf(0.975, df=df)
        VarianceValues = VarianceVector[Label]
        StdErrorVarianceVector[Label] = {
            "lower": (valid_counts - 1) * VarianceValues / upper_chi2,
            "upper": (valid_counts - 1) * VarianceValues / lower_chi2,
        }

    ### Calculate Difference (Method - Baseline) if specified ###
    if RelativeError:
        if RelativeError in MeanVector:
            Y_Label = f"Error Difference (Method - {RelativeError})"
            BaselineMean = MeanVector[RelativeError].copy()

            for Label in MeanVector:
                MeanVector[Label] = MeanVector[Label] - BaselineMean

            # Manual Clamp for 100% Labeled
            # At 100%, Method == Baseline, so Difference must be 0.0
            last_idx = len(common_grid_pct) - 1
            if common_grid_pct[last_idx] >= 100.0 - 1e-9:
                for Label in MeanVector:
                    MeanVector[Label][last_idx] = 0.0
                    StdErrorVector[Label][last_idx] = 0.0
        else:
            print(f"  > Warning: Baseline '{RelativeError}' not found. Skipping normalization.")

    ### Mean Plot ###
    fig_mean, ax_mean = plt.subplots(figsize=FigSize)

    for Label, MeanValues in MeanVector.items():
        valid_counts = ValidCounts[Label]
        plot_mask = valid_counts > 0  # don't draw where no sim has data (yet/anymore)

        x = common_grid_pct[plot_mask]
        y = MeanValues[plot_mask]
        StdErrorValues = StdErrorVector[Label][plot_mask]

        color = Colors.get(Label, None) if Colors else None
        linestyle = Linestyles.get(Label, ":") if Linestyles else ":"
        legend_label = LegendMapping.get(Label, Label) if LegendMapping else Label

        ax_mean.plot(x, y, label=legend_label, color=color, linestyle=linestyle)
        ax_mean.fill_between(
            x,
            y - CriticalValue * StdErrorValues,
            y + CriticalValue * StdErrorValues,
            alpha=TransparencyVal,
            color=color,
        )

    ax_mean.set_xlabel("Percent of Learning Pool Labeled")
    ax_mean.set_ylabel(Y_Label)

    if RelativeError:
        ax_mean.axhline(y=0.0, color="r", linestyle="-", linewidth=1, alpha=0.5)

    if show_legend:
        ax_mean.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    if isinstance(xlim, list):
        ax_mean.set_xlim(xlim)

    ### Variance Plot ###
    fig_var = None
    if VarInput:
        fig_var, ax_var = plt.subplots(figsize=FigSize)

        for Label, VarianceValues in VarianceVector.items():
            valid_counts = ValidCounts[Label]
            plot_mask = valid_counts > 0

            x = common_grid_pct[plot_mask]
            y = VarianceValues[plot_mask]

            color = Colors.get(Label, None) if Colors else None
            linestyle = Linestyles.get(Label, "-") if Linestyles else "-"
            legend_label = LegendMapping.get(Label, Label) if LegendMapping else Label

            ax_var.plot(x, y, label=legend_label, color=color, linestyle=linestyle)
            lower_bound = StdErrorVarianceVector[Label]["lower"][plot_mask]
            upper_bound = StdErrorVarianceVector[Label]["upper"][plot_mask]
            ax_var.fill_between(x, lower_bound, upper_bound, alpha=TransparencyVal, color=color)

        ax_var.set_xlabel("Percent of Learning Pool Labeled")
        ax_var.set_ylabel("Variance of " + (Y_Label if Y_Label else "Error"))
        ax_var.legend(loc="upper right")
        if isinstance(xlim, list):
            ax_var.set_xlim(xlim)

    return (fig_mean, fig_var)


### Main Wrapper Function ###
def generate_all_plots(aggregated_results_dir, image_dir, show_legend=True, single_dataset=None):
    """
    Wrapper function to load aggregated .pkl files and generates all specified plots.
    Can process all datasets or just a single one if specified.
    """

    ### Aesthetics and Plot Definitions ###
    master_colors = {
        "Passive Learning": "gray",
        "GSx": "cornflowerblue",
        "GSy": "salmon",
        "iGS": "red",
        "WiGS (Static w_x=0.75)": "lightgreen",
        "WiGS (Static w_x=0.5)": "forestgreen",
        "WiGS (Static w_x=0.25)": "darkgreen",
        "WiGS (Time-Decay, Linear)": "orange",
        "WiGS (Time-Decay, Exponential)": "saddlebrown",
        "WiGS (MAB-UCB1, c=0.5)": "orchid",
        "WiGS (MAB-UCB1, c=2.0)": "darkviolet",
        "WiGS (MAB-UCB1, c=5.0)": "indigo",
        "WiGS (SAC)": "darkcyan",
        "QBC": "goldenrod",
    }
    master_linestyles = {
        "Passive Learning": ":",
        "GSx": ":",
        "GSy": ":",
        "iGS": "-",
        "WiGS (Static w_x=0.75)": "-.",
        "WiGS (Static w_x=0.5)": "-.",
        "WiGS (Static w_x=0.25)": "-.",
        "WiGS (Time-Decay, Linear)": "-.",
        "WiGS (Time-Decay, Exponential)": "-.",
        "WiGS (MAB-UCB1, c=0.5)": "-.",
        "WiGS (MAB-UCB1, c=2.0)": "-.",
        "WiGS (MAB-UCB1, c=5.0)": "-.",
        "WiGS (SAC)": "-",
        "QBC": "-.",
    }
    master_legend = {
        "Passive Learning": "Random",
        "GSx": "GSx",
        "GSy": "GSy",
        "iGS": "iGS",
        "WiGS (Static w_x=0.75)": "WiGS (Static, w_x=0.75)",
        "WiGS (Static w_x=0.5)": "WiGS (Static, w_x=0.5)",
        "WiGS (Static w_x=0.25)": "WiGS (Static, w_x=0.25)",
        "WiGS (Time-Decay, Linear)": "WiGS (Linear Decay)",
        "WiGS (Time-Decay, Exponential)": "WiGS (Exponential Decay)",
        "WiGS (MAB-UCB1, c=2.0)": "MAB-UCB1, c=2.0",
        "WiGS (MAB-UCB1, c=5.0)": "MAB-UCB1, c=5.0",
        "WiGS (SAC)": "WiGS (SAC)",
        "QBC": "QBC",
    }

    ### Set up ###
    metrics_to_plot = ["RMSE", "MAE", "R2", "CC"]
    plot_types = {
        "trace": None,
        "trace_relative_random": "Passive Learning",
        "trace_relative_iGS": "iGS",
    }
    eval_types = ["full_pool", "full_test", "train"]
    strategies_to_exclude = {"WiGS (Static w_x=0.5)", "WiGS (MAB-UCB1, c=0.5)"}
    total_pool_size = None

    ### Dynamically find datasets ###
    if single_dataset:
        dataset_folders = [single_dataset]
        print(f"--- Starting Plot Generation for single dataset: {single_dataset} ---")
    else:
        print("--- Starting Plot Generation from Aggregated Results ---")
        dataset_folders = [
            d
            for d in os.listdir(aggregated_results_dir)
            if os.path.isdir(os.path.join(aggregated_results_dir, d))
        ]

    total_datasets = len(dataset_folders)

    for i, data_name in enumerate(dataset_folders):
        print(f"\n({i+1}/{total_datasets}) Processing dataset: {data_name}...")
        dataset_path = os.path.join(aggregated_results_dir, data_name)

        for eval_type in eval_types:
            print(f"  > Generating plots for '{eval_type}' metrics...")
            eval_metric_path = os.path.join(dataset_path, f"{eval_type}_metrics")

            if not os.path.isdir(eval_metric_path):
                print(f"    - Skipping: Directory not found at {eval_metric_path}")
                continue

            for metric in metrics_to_plot:
                metric_pkl_path = os.path.join(eval_metric_path, f"{metric}.pkl")

                if not os.path.exists(metric_pkl_path):
                    continue

                with open(metric_pkl_path, "rb") as f:
                    results_for_metric = pickle.load(f)

                print(f"    - results_for_metric  {metric} for {eval_type}")

                # Indices #
                indices_file_path = os.path.join(dataset_path, "InitialIndices.csv")
                try:
                    if not os.path.exists(indices_file_path):
                        raise FileNotFoundError

                    indices_df = pd.read_csv(indices_file_path)
                    initial_train_size = len(indices_df)
                    if initial_train_size == 0:
                        raise ValueError("InitialIndices.csv is empty.")
                except FileNotFoundError:
                    print(
                        f"  > Warning: InitialIndices.csv not found for {data_name} at {indices_file_path}. Skipping {metric} plot."
                    )
                    continue
                except ValueError as e:
                    print(
                        f"  > Warning: Error reading InitialIndices.csv for {data_name}: {e}. Skipping {metric} plot."
                    )
                    continue
                except Exception as e:
                    print(
                        f"  > Warning: An unexpected error occurred loading InitialIndices.csv for {data_name}: {e}. Skipping {metric} plot."
                    )
                    continue

                # TotalPoolSize #
                total_pool_size_path = os.path.join(dataset_path, "TotalPoolSize.csv")

                if os.path.exists(total_pool_size_path):
                    total_pool_size_df = pd.read_csv(total_pool_size_path, index_col="Simulation")
                    # The CSV has one row per sim and one column per strategy
                    # (NaN-padded for strategies with fewer sims). We only need
                    # ONE pool size per strategy for the % axis, so take the
                    # first non-NaN value per column -- but warn if the sims of
                    # a strategy actually disagree, since that would mean the
                    # single-value-per-Label assumption downstream is wrong.
                    total_pool_size = {}
                    for strategy in results_for_metric:
                        if strategy not in total_pool_size_df.columns:
                            continue
                        values = total_pool_size_df[strategy].dropna().tolist()
                        if not values:
                            continue
                        if len(set(values)) > 1:
                            print(
                                f"  > Warning: TotalPoolSize differs across sims for "
                                f"'{strategy}' ({set(values)}); using the first value "
                                f"({values[0]})."
                            )
                        total_pool_size[strategy] = values[0]
                else:
                    total_pool_size = None

                # SimulationParameters #
                simulation_parameters = os.path.join(dataset_path, "SimulationParameters.yaml")
                try:
                    with open(simulation_parameters, "r") as file:
                        params = yaml.safe_load(file)
                except FileNotFoundError:
                    print(f"Error : File {simulation_parameters} not found.")

                # read ErrorVecs_iteration.csv
                try:
                    error_vecs_path = os.path.join(dataset_path, "ErrorVecs_iteration.csv")
                except FileNotFoundError:
                    print(f"Error : File {error_vecs_path} not found.")
                df = pd.read_csv(error_vecs_path, index_col=0)
                error_vecs_iteration = {}
                for col in df.columns:
                    error_vecs_iteration[col] = df[col].apply(safe_literal_eval).tolist()

                sim_name = {}
                for strategy, df in results_for_metric.items():
                    sim_name[strategy] = {}
                    for param in params[strategy]:
                        sim_name[strategy][param["Sim"]] = (
                            f"sim{param['Sim'][4:]}."
                            f"s{param['Seed']}."
                            f"ncv{flag_to_int(param.get('no_cv', False))}"
                            f".ep{param['hl_max_epoch']}."
                            f"kt{param['k_top_candidate']}"
                            f".prs{param['subset_rand_candidat']}."
                            f"cur{flag_to_int(param.get('curriculum', False))}"
                        )

                try:
                    # get k_top
                    selection_history_dir = os.path.join(dataset_path, "selection_history")
                    k_top = {}
                    for strategy, df in results_for_metric.items():

                        k_top_path = os.path.join(
                            selection_history_dir, f"{strategy}_SelectionHistory.csv"
                        )

                        if os.path.exists(k_top_path):
                            # k_top[strategy] = pd.read_csv(k_top_path)

                            df = pd.read_csv(
                                k_top_path, index_col="Iteration"
                            )  # adapte si index_label différent
                            # Reconvertir chaque cellule "[16899, 623, ...]" en vraie liste Python
                            df = df.fillna("[]")

                            df = df.map(ast.literal_eval)
                            k_top[strategy] = df

                            print(f"  > Using k_top = {list(k_top.keys())} for dataset {data_name}")
                        else:
                            k_top = 1  # Value by default (ex: beer)
                            print(
                                f"> Warning: {strategy}_SelectionHistory.csv not found for {data_name}"
                            )
                except:
                    print("selection_history_dir fail")

                # Filter out the excluded strategies
                filtered_results = {
                    strategy: df
                    for strategy, df in results_for_metric.items()
                    if strategy not in strategies_to_exclude
                }

                for folder_name, baseline in plot_types.items():
                    y_label = f"Normalized {metric}" if baseline else metric
                    subtitle = f"Performance ({eval_type.capitalize()} {metric}) on {data_name.upper()} Dataset"

                    print(f"\t *{y_label}, {subtitle} ")

                    output_eval_name = "trace_plots" if eval_type == "full_pool" else eval_type
                    base_plot_path = os.path.join(image_dir, output_eval_name, metric, folder_name)
                    os.makedirs(os.path.join(base_plot_path, "trace"), exist_ok=True)
                    os.makedirs(os.path.join(base_plot_path, "variance"), exist_ok=True)

                    try:

                        sim_ids = {
                            strategy: [p["Sim"] for p in params[strategy]]
                            for strategy in results_for_metric
                        }

                        indiv_plots = IndividualTracesPlot(
                            RelativeError=baseline,
                            # Colors=master_colors,
                            LegendMapping=master_legend,
                            Linestyles=master_linestyles,
                            Y_Label=y_label,
                            Subtitle=subtitle,
                            TransparencyVal=1.0,
                            VarInput=True,
                            CriticalValue=1.96,
                            initial_train_size=initial_train_size,
                            k_top=k_top,
                            show_legend=show_legend,
                            total_pool_size=total_pool_size,
                            sim_name=sim_name,
                            sim_ids=sim_ids,
                            **filtered_results,
                        )

                        all_plot_path = os.path.join(
                            base_plot_path, "trace", f"{data_name}_{metric}_allplot.png"
                        )
                        indiv_plots.savefig(all_plot_path, bbox_inches="tight", dpi=300)
                        plt.close(indiv_plots)

                        TracePlotMean, TracePlotVariance = MeanVariancePlot(
                            RelativeError=baseline,
                            Colors=master_colors,
                            LegendMapping=master_legend,
                            Linestyles=master_linestyles,
                            Y_Label=y_label,
                            Subtitle=subtitle,
                            TransparencyVal=0.1,
                            VarInput=True,
                            CriticalValue=1.96,
                            initial_train_size=initial_train_size,
                            k_top=k_top,
                            show_legend=show_legend,
                            total_pool_size=total_pool_size,
                            sim_name=sim_name,
                            **filtered_results,
                        )

                        trace_plot_path = os.path.join(
                            base_plot_path, "trace", f"{data_name}_{metric}_TracePlot.png"
                        )
                        TracePlotMean.savefig(trace_plot_path, bbox_inches="tight", dpi=300)
                        plt.close(TracePlotMean)

                        if TracePlotVariance:
                            variance_plot_path = os.path.join(
                                base_plot_path,
                                "variance",
                                f"{data_name}_{metric}_VariancePlot.png",
                            )
                            TracePlotVariance.savefig(
                                variance_plot_path, bbox_inches="tight", dpi=300
                            )
                            plt.close(TracePlotVariance)

                    except Exception as e:
                        print(f"Error in MeanVariancePlot : {e}")
                        print("\nStacktrace all :")
                        traceback.print_exc()

        print(f"Finished all plots for {data_name}.")
    print("\n--- Plot Generation Complete ---")


### GENERATE LEGEND ###
def generate_legend(legend_mapping, colors, linestyles, output_path, ncol=4):
    """
    Generates a standalone legend image from the master style dictionaries.
    """

    # Create dummy plot handles for the legend
    handles = []
    labels = []

    for long_name, short_name in legend_mapping.items():
        color = colors.get(long_name)
        ls = linestyles.get(long_name, "-")

        if color is None:
            continue

        # Create a dummy line object
        line = plt.Line2D([0], [0], color=color, linestyle=ls, label=short_name)
        handles.append(line)
        labels.append(short_name)

    # Figure height needs to be slightly taller for 3 rows
    fig = plt.figure(figsize=(16, 3))

    # Create the legend
    fig_legend = fig.legend(handles, labels, loc="center", frameon=True, ncol=ncol)
    plt.gca().axis("off")
    fig.savefig(
        output_path,
        bbox_inches=fig_legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted()),
        dpi=300,
        transparent=True,
    )
    plt.close(fig)
    print(f"--- Legend generation complete ---")


### MAIN ###
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate plots for simulation results.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Optional: name of a single dataset folder to process.",
    )
    parser.add_argument(
        "--no-legend",
        dest="show_legend",
        action="store_false",
        help="Disable legends on individual plots (for later compilation).",
    )
    parser.add_argument(
        "--legend_only",
        action="store_true",
        help="If set, only generate a standalone legend file and exit.",
    )
    args = parser.parse_args()

    ## Define Paths ##
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    except NameError:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))

    AGGREGATED_RESULTS_DIR = os.path.join(
        PROJECT_ROOT, "Results", "simulation_results", "aggregated"
    )
    IMAGE_DIR = os.path.join(PROJECT_ROOT, "Results", "images")

    if args.legend_only:
        master_colors = {
            "Passive Learning": "gray",
            "GSx": "cornflowerblue",
            "GSy": "salmon",
            "iGS": "red",
            "WiGS (Static w_x=0.75)": "lightgreen",
            "WiGS (Static w_x=0.5)": "forestgreen",
            "WiGS (Static w_x=0.25)": "darkgreen",
            "WiGS (Time-Decay, Linear)": "orange",
            "WiGS (Time-Decay, Exponential)": "saddlebrown",
            "WiGS (MAB-UCB1, c=0.5)": "orchid",
            "WiGS (MAB-UCB1, c=2.0)": "darkviolet",
            "WiGS (MAB-UCB1, c=5.0)": "indigo",
            "WiGS (SAC)": "darkcyan",
            "QBC": "goldenrod",
        }

        master_linestyles = {
            "Passive Learning": ":",
            "GSx": ":",
            "GSy": ":",
            "iGS": "-",
            "WiGS (Static w_x=0.75)": "-.",
            "WiGS (Static w_x=0.5)": "-.",
            "WiGS (Static w_x=0.25)": "-.",
            "WiGS (Time-Decay, Linear)": "-.",
            "WiGS (Time-Decay, Exponential)": "-.",
            "WiGS (MAB-UCB1, c=0.5)": "-.",
            "WiGS (MAB-UCB1, c=2.0)": "-.",
            "WiGS (MAB-UCB1, c=5.0)": "-.",
            "WiGS (SAC)": "-",
            "QBC": "-.",
        }

        master_legend = {
            "Passive Learning": "Random",
            "GSx": "GSx",
            "GSy": "GSy",
            "iGS": "iGS",
            "WiGS (Static w_x=0.75)": "WiGS (Static, w_x=0.75)",
            "WiGS (Static w_x=0.5)": "WiGS (Static, w_x=0.5)",
            "WiGS (Static w_x=0.25)": "WiGS (Static, w_x=0.25)",
            "WiGS (Time-Decay, Linear)": "WiGS (Linear Decay)",
            "WiGS (Time-Decay, Exponential)": "WiGS (Exponential Decay)",
            "WiGS (MAB-UCB1, c=0.5)": "WiGS (MAB, c=0.5)",
            "WiGS (MAB-UCB1, c=2.0)": "WiGS (MAB, c=2.0)",
            "WiGS (MAB-UCB1, c=5.0)": "WiGS (MAB, c=5.0)",
            "WiGS (SAC)": "WiGS (SAC)",
            "QBC": "QBC",
        }

        # Define strategies to *exclude* from the legend #
        strategies_to_exclude = {
            "WiGS (Static w_x=0.5)",
            # 'WiGS (MAB-UCB1, c=0.5)',
            "WiGS (MAB-UCB1, c=2.0)",
        }

        # Filter the master legend #
        filtered_legend_mapping = {
            long: short
            for long, short in master_legend.items()
            if long not in strategies_to_exclude
        }

        # Define the output path #
        legend_output_path = os.path.join(IMAGE_DIR, "benchmark_legend.png")

        # Generate the legend #
        generate_legend(
            legend_mapping=filtered_legend_mapping,
            colors=master_colors,
            linestyles=master_linestyles,
            output_path=legend_output_path,
            ncol=6,
        )

    else:
        ## Execute the main plotting function ##
        generate_all_plots(
            aggregated_results_dir=AGGREGATED_RESULTS_DIR,
            image_dir=IMAGE_DIR,
            show_legend=args.show_legend,
            single_dataset=args.dataset,
        )
