"""Plotting script for the ROOT vs Parquet ingestion benchmark.

Run after generating the benchmark JSON(s):

    pytest tests/benchmarks/test_root_vs_parquet.py --benchmark-only -s -m "not slow"
    pytest tests/benchmarks/test_root_vs_parquet.py --benchmark-only -s -m slow

    python tests/benchmarks/plot_root_vs_parquet.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import mplhep
import numpy as np
import pandas as pd

FILE_TYPES = ["parquet", "root", "rdataloader"]
# Pretty x-axis labels; "rdataloader" reads the same .root files as "root", just via PyROOT.
FILE_TYPE_LABELS = {
    "parquet": "parquet\n(dask)",
    "root": "root\n(uproot)",
    "rdataloader": "root\n(RDataLoader)",
}
COMPONENTS = ["Graph Building", "Column-based Iteration", "Row-based Iteration"]
TEST_METHODS = ["only_metadata", "materialize_partitions", "iterate_dataloader"]
COLORS = ["lightcoral", "lightgreen", "lightblue"]
RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"
DEFAULT_OUTPUT = PLOTS_DIR / "ingestion_decomposed.pdf"
INPUT_FILES = [RESULTS_DIR / "root_vs_parquet_fast.json", RESULTS_DIR / "root_vs_parquet_slow.json"]

# Description of the files, drawn inside the axes. Edit when pointing at a different dataset.
# Current values: the 21-file Delphes set (443 branches, 21 GB as .root / 20 GB as .parquet),
# of which the benchmark reads 7 feature columns + 1 label over all 210k events.
ANNOTATION = "Files: 443 columns, 21GB\nRead: 8 columns, 210k events"


def load_benchmark_json(path: Union[str, Path], merge_index: bool = False) -> pd.DataFrame:
    """Load a pytest-benchmark autosave JSON file into a flat DataFrame.

    Args:
        path: Path to the `.json` file autosaved by pytest-benchmark (`--benchmark-autosave`).
        merge_index: If True, set/sort the index to
            `["file_type", "column_mode", "file_percentage", "num_events"]`.

    Returns:
        pd.DataFrame: One row per benchmarked parametrization, with `mean_time`, `median_time`,
            `min_time`, `max_time`, `stddev` and `rounds` extracted from the benchmark stats.
    """
    with open(path) as f:
        data = json.load(f)

    rows = []
    for b in data["benchmarks"]:
        params = b["params"]
        stats = b["stats"]
        rows.append(
            {
                "name": b["name"],
                "file_type": params["file_type"],
                "num_events": params["num_events"],
                "test_method": params["test_method"],
                "column_mode": params["column_mode"],
                "file_percentage": params["file_percentage"],
                "mean_time": stats["mean"],
                "median_time": stats["median"],
                "min_time": stats["min"],
                "max_time": stats["max"],
                "stddev": stats["stddev"],
                "rounds": stats["rounds"],
            }
        )

    df = pd.DataFrame(rows)

    if merge_index:
        df = df.set_index(["file_type", "column_mode", "file_percentage", "num_events"]).sort_index()

    return df


def load_default_benchmarks(merge_index: bool = False) -> pd.DataFrame:
    """Load and concatenate the fast/slow benchmark JSONs written by `tests/benchmarks/conftest.py`.

    Args:
        merge_index: If True, set/sort the index to
            `["file_type", "column_mode", "file_percentage", "num_events"]`.

    Returns:
        pd.DataFrame: Rows from every existing, non-empty file in `INPUT_FILES`.

    Raises:
        FileNotFoundError: If none of `INPUT_FILES` exist yet.
    """
    frames = [load_benchmark_json(p) for p in INPUT_FILES if p.exists() and p.stat().st_size > 0]
    if not frames:
        raise FileNotFoundError(
            f"No benchmark JSON found in {[str(p) for p in INPUT_FILES]}. Run "
            '`pytest tests/benchmarks/test_root_vs_parquet.py --benchmark-only -s -m "not slow"` '
            "(and/or `-m slow`) first."
        )
    df = pd.concat(frames, ignore_index=True)
    if merge_index:
        df = df.set_index(["file_type", "column_mode", "file_percentage", "num_events"]).sort_index()
    return df


def select_benchmarks(
    df: pd.DataFrame,
    column_mode: str = "config",
    file_percentage: float = 0.0,
    num_events: int = 1000,
) -> pd.Series:
    """Filter to a single `(column_mode, file_percentage, num_events)` slice and group by
    `(file_type, test_method)`, averaging `mean_time` over rounds.

    Args:
        df: Flat DataFrame as returned by `load_benchmark_json`.
        column_mode: Which `column_mode` parametrization to select (`"one"` or `"config"`).
        file_percentage: Which `file_percentage` parametrization to select.
        num_events: Which `num_events` parametrization to select.

    Returns:
        pd.Series: `mean_time` indexed by `(file_type, test_method)`, matching the `grouped`
            object expected by `plot_root_vs_parquet`.

    Raises:
        ValueError: If no rows match the requested filter.
    """
    mask = (
        (df["column_mode"] == column_mode)
        & (df["file_percentage"] == file_percentage)
        & (df["num_events"] == num_events)
    )
    subset = df[mask]
    if subset.empty:
        available = df[["column_mode", "file_percentage", "num_events"]].drop_duplicates()
        raise ValueError(
            f"No benchmark rows found for column_mode={column_mode!r}, "
            f"file_percentage={file_percentage!r}, num_events={num_events!r}.\n"
            f"Available combinations:\n{available.to_string(index=False)}"
        )
    return subset.groupby(["file_type", "test_method"])["mean_time"].mean()


def plot_root_vs_parquet(
    grouped: pd.Series,
    output_path: Union[str, Path],
    annotation: Optional[str] = None,
) -> plt.Figure:
    """Build the Graph Building / Column-based Iteration / Row-based Iteration comparison chart.

    Args:
        grouped: `mean_time` indexed by `(file_type, test_method)`, as returned by
            `select_benchmarks`.
        output_path: Where to save the figure. Parent directories are created if needed.
        annotation: Optional text box

    Returns:
        matplotlib.figure.Figure: The created figure (caller is responsible for `plt.close(fig)`).
    """
    # Only plot backends actually present: the rdataloader arm is skipped when PyROOT is
    # unavailable and must not blow up the plot when missing.
    present = grouped.index.get_level_values("file_type")
    file_types = [ft for ft in FILE_TYPES if ft in present]
    if not file_types:
        raise ValueError(f"None of {FILE_TYPES} are present in the benchmark data.")

    times = {}
    for ft in file_types:
        graph_building = grouped.loc[ft, "only_metadata"]
        materialize = grouped.loc[ft, "materialize_partitions"]
        iterate = grouped.loc[ft, "iterate_dataloader"]

        if materialize < iterate:
            # Nested: materialization is a genuine sub-phase of the full iteration (the dask arms
            # materialize each partition, then walk its rows), so the stages subtract.
            materialization = materialize - graph_building
            total = iterate - materialize
        else:
            # NOT nested. RDataLoader's AsNumpy() is a separate code path, not a sub-phase of its
            # streaming iteration, so `iterate - materialize` is meaningless (often negative).
            # Draw it as a standalone reference and measure row iteration from graph building --
            # so this backend's bars do NOT sum to its total.
            materialization = materialize - graph_building
            total = iterate - graph_building
            print(
                f"[note] {ft}: materialize ({materialize:.3f}s) >= iterate ({iterate:.3f}s), so the "
                "stages are not nested. Its 'Column-based Iteration' bar is a standalone reference "
                "(a different code path) and its bars do not sum to its end-to-end time."
            )

        times[ft] = [graph_building, max(0.0, materialization), max(0.0, total)]

    x = np.arange(len(file_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(2.4 * len(file_types) + 1.6, 4.6), dpi=600)
    for i, comp in enumerate(COMPONENTS):
        ax.bar(
            x + i * width,
            [times[ft][i] for ft in file_types],
            width,
            label=comp,
            color=COLORS[i],
            alpha=1,
            zorder=2,
        )
        for j, ft in enumerate(file_types):
            bar_x = x[j] + i * width
            bar_height = times[ft][i]
            unit = "ms" if bar_height < 1 else "s"
            value = bar_height * 1000 if unit == "ms" else bar_height
            ax.text(bar_x, bar_height + 0.01, f"{value:.1f}{unit}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Ingestion backend")
    ax.set_ylabel("Average Time [s]")
    ax.set_xticks(x + width)
    ax.set_xticklabels([FILE_TYPE_LABELS.get(ft, ft) for ft in file_types])
    max_height = max(v for values in times.values() for v in values)
    ax.set_ylim(top=max_height * 1.35)
    ax.legend(loc="upper left")

    if annotation:
        ax.text(
            0.02,
            0.65,
            annotation,
            transform=ax.transAxes,
            fontsize=10,
            ha="left",
            va="bottom",
        )

    mplhep.label.exp_label(loc=0, exp="NEEDLE", ax=ax, rlabel="")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)

    return fig


def main(argv: Optional[list] = None) -> Path:
    """CLI entry point. Loads a benchmark JSON, filters it, plots it and saves the figure.

    Args:
        argv: Optional argument list (for testing). Defaults to `sys.argv[1:]`.

    Returns:
        Path: The path the figure was saved to.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=f"Path to a pytest-benchmark JSON file. Defaults to {[str(p) for p in INPUT_FILES]}.",
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Where to save the plot.")
    parser.add_argument("--column-mode", type=str, default="config", choices=["one", "config"],
                        help="Which column_mode slice to plot.")
    parser.add_argument("--file-percentage", type=float, default=0.0,
                        help="Which file_percentage slice to plot. Use 100 for the full file list.")
    parser.add_argument("--num-events", type=int, default=1000,
                        help="Which num_events slice to plot. Use -1 for the 'all events' parametrization.")
    parser.add_argument("--annotation", type=str, default=ANNOTATION,
                        help="Text box drawn on the plot. Defaults to the ANNOTATION constant.")
    args = parser.parse_args(argv)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_benchmark_json(Path(args.input)) if args.input else load_default_benchmarks()
    grouped = select_benchmarks(
        df,
        column_mode=args.column_mode,
        file_percentage=args.file_percentage,
        num_events=args.num_events,
    )
    annotation = args.annotation
    fig = plot_root_vs_parquet(grouped, args.output, annotation=annotation)
    plt.close(fig)

    source = args.input if args.input else [str(p) for p in INPUT_FILES if p.exists()]
    print(f"Saved plot to {args.output} (source: {source})")
    return Path(args.output)


if __name__ == "__main__":
    main()
