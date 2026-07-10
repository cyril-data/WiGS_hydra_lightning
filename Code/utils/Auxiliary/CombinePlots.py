### combine_plots.py ###
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

CRITICAL_VALUE = 1.96


def load_all_plot_data(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def plot_combined(
    all_plot_data,
    selections,
    output_path,
    y_label="Metric",
    title=None,
    show_ci=True,
    figsize=(9, 6),
):
    """
    selections: liste de dicts, chacun décrivant UNE courbe à tracer :
        {
            "dataset": "beer",
            "eval_type": "full_pool",
            "metric": "CC",
            "plot_type": "trace",
            "strategy": "iGS",
            "label": "Beer - iGS",   # optionnel, sinon auto-généré
            "color": "red",          # optionnel
            "linestyle": "-",        # optionnel
        }
    """
    fig, ax = plt.subplots(figsize=figsize)

    for sel in selections:
        try:
            data_out = all_plot_data[sel["dataset"]][sel["eval_type"]][sel["metric"]][
                sel["plot_type"]
            ]
        except KeyError:
            print(f"  > Warning: combinaison introuvable pour {sel}")
            continue

        strategy = sel["strategy"]
        if strategy not in data_out["mean"]:
            print(f"  > Warning: strategy '{strategy}' absente de {sel}")
            continue

        x = data_out["x"][strategy]
        mean = data_out["mean"][strategy]
        stderr = data_out["stderr"][strategy]

        label = (
            sel.get("label")
            or f"{sel['dataset']} | {strategy} | {sel['metric']} ({sel['plot_type']})"
        )
        color = sel.get("color")
        linestyle = sel.get("linestyle", "-")

        ax.plot(x, mean, label=label, color=color, linestyle=linestyle)
        if show_ci:
            ax.fill_between(
                x,
                mean - CRITICAL_VALUE * stderr,
                mean + CRITICAL_VALUE * stderr,
                alpha=0.15,
                color=color,
            )

    ax.set_xlabel("Percent of Learning Pool Labeled")
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"--- Saved combined plot to {output_path} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple traces on a single plot.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to all_plot_data.pkl")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    args = parser.parse_args()

    all_plot_data = load_all_plot_data(args.data_path)

    # --- EXEMPLE : comparer iGS sur plusieurs datasets, en full_pool, pour CC, en trace normal ---
    selections = [
        {
            "dataset": "beer",
            "eval_type": "full_pool",
            "metric": "CC",
            "plot_type": "trace",
            "strategy": "iGS",
            "color": "red",
        },
        {
            "dataset": "wine",
            "eval_type": "full_pool",
            "metric": "CC",
            "plot_type": "trace",
            "strategy": "iGS",
            "color": "blue",
        },
        {
            "dataset": "housing",
            "eval_type": "full_pool",
            "metric": "CC",
            "plot_type": "trace",
            "strategy": "iGS",
            "color": "green",
        },
    ]

    plot_combined(
        all_plot_data,
        selections,
        output_path=args.output,
        y_label="CC",
        title="iGS Performance Across Datasets",
    )
