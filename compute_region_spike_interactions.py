from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# This script now computes two related summaries from 100 ms binned spike trains:
# 1. Population correlation between brain regions: correlation between region-level spike-count traces.
# 2. Mean neuron-pair correlation: average correlation across all neuron pairs spanning two regions.
#
# Within-region correlations are handled by taking all unique neuron pairs inside one region.
# For this within-region case, we do not use the population correlation on the diagonal because
# a region correlated with itself would trivially be 1.0 and would not reflect neuron-to-neuron coupling.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute 100 ms spike-train interaction scores between brain regions "
            "for each mouse folder and save summary plots."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("D:/PreCosyneBrainhack"),
        help="Root directory containing mouse folders such as 2, 3, ..., 18.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where result tables will be written. Defaults to "
            "<dataset-root>/output/region_interactions_100ms."
        ),
    )
    parser.add_argument(
        "--mouse-ids",
        type=int,
        nargs="+",
        default=list(range(2, 19)),
        help="Mouse folder names to analyze.",
    )
    parser.add_argument(
        "--bin-size-ms",
        type=float,
        default=100.0,
        help="Bin width in milliseconds for the spike-count time series.",
    )
    # Kept for backward compatibility, but within-region correlation is now on by default.
    parser.add_argument(
        "--include-within-region",
        dest="include_within_region",
        action="store_true",
        default=True,
        help="Compute within-region neuron-pair correlations. This is on by default.",
    )
    parser.add_argument(
        "--skip-within-region",
        dest="include_within_region",
        action="store_false",
        help="Skip within-region neuron-pair correlations.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving PNG plots and only export CSV files.",
    )
    return parser.parse_args()


def format_region_label(region: object) -> str:
    if isinstance(region, (bytes, np.bytes_)):
        return region.decode("utf-8")
    return str(region)


def load_mouse_clusters(mouse_folder: Path) -> pd.DataFrame:
    spikes = np.load(mouse_folder / "spikes.npy")
    clusters = np.load(mouse_folder / "clusters.npy")
    brain_area_info = np.load(mouse_folder / "brain_area.npy", allow_pickle=True).item()

    region_map = dict(zip(brain_area_info["cluster_id"], brain_area_info["brain_area"]))

    cluster_spikes: dict[int, list[float]] = {}
    for cluster_id, spike_time in zip(clusters, spikes):
        cluster_spikes.setdefault(int(cluster_id), []).append(float(spike_time))

    rows: list[dict[str, object]] = []
    for cluster_id, spike_times in cluster_spikes.items():
        region = region_map.get(cluster_id)
        if region is None:
            continue
        rows.append(
            {
                "cluster_id": cluster_id,
                "brain_region": format_region_label(region),
                "spike_times": np.asarray(spike_times, dtype=float),
            }
        )

    if not rows:
        raise ValueError(f"No cluster-to-region assignments found in {mouse_folder}")

    return pd.DataFrame(rows).sort_values(["brain_region", "cluster_id"]).reset_index(drop=True)


def build_binned_matrix(
    spike_trains: list[np.ndarray],
    bin_size_s: float,
    recording_start: float,
    recording_end: float,
) -> np.ndarray:
    n_bins = int(np.ceil((recording_end - recording_start) / bin_size_s))
    if n_bins < 2:
        raise ValueError("Recording is too short to compute an interaction score.")

    binned = np.zeros((len(spike_trains), n_bins), dtype=float)
    for row_idx, spike_times in enumerate(spike_trains):
        bin_ids = np.floor((spike_times - recording_start) / bin_size_s).astype(int)
        valid = (bin_ids >= 0) & (bin_ids < n_bins)
        if np.any(valid):
            counts = np.bincount(bin_ids[valid], minlength=n_bins)
            binned[row_idx, :] = counts[:n_bins]

    return binned


def correlation_or_nan(trace_a: np.ndarray, trace_b: np.ndarray) -> float:
    if trace_a.size != trace_b.size:
        raise ValueError("Time series must have the same number of bins.")
    if np.std(trace_a) == 0 or np.std(trace_b) == 0:
        return np.nan
    return float(np.corrcoef(trace_a, trace_b)[0, 1])


def build_symmetric_matrix(result_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    regions = sorted(set(result_df["region_a"]).union(result_df["region_b"]))
    matrix_df = pd.DataFrame(np.nan, index=regions, columns=regions)

    for row in result_df.itertuples(index=False):
        value = getattr(row, value_col)
        matrix_df.loc[row.region_a, row.region_b] = value
        matrix_df.loc[row.region_b, row.region_a] = value

    return matrix_df


def pairwise_interactions(
    mouse_df: pd.DataFrame,
    bin_size_s: float,
    include_within_region: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_spikes = np.concatenate(mouse_df["spike_times"].to_numpy())
    recording_start = 0.0 if np.nanmin(all_spikes) >= 0 else float(np.nanmin(all_spikes))
    recording_end = float(np.nanmax(all_spikes)) + bin_size_s

    binned = build_binned_matrix(
        spike_trains=mouse_df["spike_times"].tolist(),
        bin_size_s=bin_size_s,
        recording_start=recording_start,
        recording_end=recording_end,
    )

    regions = sorted(mouse_df["brain_region"].dropna().unique().tolist())
    results: list[dict[str, object]] = []

    for region_a, region_b in combinations(regions, 2):
        results.extend(
            summarize_region_pair(mouse_df, binned, region_a, region_b, bin_size_s)
        )

    if include_within_region:
        for region in regions:
            results.extend(
                summarize_region_pair(mouse_df, binned, region, region, bin_size_s)
            )

    result_df = pd.DataFrame(results)
    if result_df.empty:
        raise ValueError("No region pairs produced an interaction table.")

    population_matrix = build_symmetric_matrix(result_df, value_col="population_corr")
    pairwise_matrix = build_symmetric_matrix(result_df, value_col="mean_pairwise_corr")
    return result_df, population_matrix, pairwise_matrix


def summarize_region_pair(
    mouse_df: pd.DataFrame,
    binned: np.ndarray,
    region_a: str,
    region_b: str,
    bin_size_s: float,
) -> list[dict[str, object]]:
    idx_a = mouse_df.index[mouse_df["brain_region"] == region_a].to_numpy()
    idx_b = mouse_df.index[mouse_df["brain_region"] == region_b].to_numpy()

    if idx_a.size == 0 or idx_b.size == 0:
        return []

    correlations: list[float] = []
    if region_a == region_b:
        pair_iter = combinations(idx_a.tolist(), 2)
    else:
        pair_iter = ((i, j) for i in idx_a for j in idx_b)

    for i, j in pair_iter:
        corr = correlation_or_nan(binned[i], binned[j])
        if not np.isnan(corr):
            correlations.append(corr)

    # Across-region population correlation is informative, but for within-region pairs it would
    # trivially be the correlation of a region with itself. We leave the diagonal of the
    # population matrix blank (NaN) and use mean_pairwise_corr for within-region summaries.
    if region_a == region_b:
        population_corr = np.nan
    else:
        pop_trace_a = binned[idx_a].sum(axis=0)
        pop_trace_b = binned[idx_b].sum(axis=0)
        population_corr = correlation_or_nan(pop_trace_a, pop_trace_b)

    return [
        {
            "region_a": region_a,
            "region_b": region_b,
            "bin_size_ms": int(round(bin_size_s * 1000)),
            "n_neurons_a": int(idx_a.size),
            "n_neurons_b": int(idx_b.size),
            "n_neuron_pairs": int(len(correlations)),
            "mean_pairwise_corr": float(np.mean(correlations)) if correlations else np.nan,
            "median_pairwise_corr": float(np.median(correlations)) if correlations else np.nan,
            "population_corr": population_corr,
        }
    ]


def save_mouse_results(
    mouse_id: int,
    interaction_df: pd.DataFrame,
    population_matrix: pd.DataFrame,
    pairwise_matrix: pd.DataFrame,
    output_dir: Path,
) -> None:
    interaction_df.to_csv(output_dir / f"mouse_{mouse_id:02d}_region_interactions.csv", index=False)
    population_matrix.to_csv(output_dir / f"mouse_{mouse_id:02d}_population_corr_matrix.csv")
    pairwise_matrix.to_csv(output_dir / f"mouse_{mouse_id:02d}_pairwise_corr_matrix.csv")


def plot_heatmap(
    matrix_df: pd.DataFrame,
    title: str,
    output_path: Path,
    cbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Brain region")
    ax.set_ylabel("Brain region")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mouse_population_heatmap(mouse_id: int, matrix_df: pd.DataFrame, output_dir: Path, bin_size_ms: float) -> None:
    plot_heatmap(
        matrix_df=matrix_df,
        title=f"Mouse {mouse_id}: population correlation ({bin_size_ms:.0f} ms bins)",
        output_path=output_dir / f"mouse_{mouse_id:02d}_population_corr_heatmap.png",
        cbar_label="Pearson r",
    )


def plot_mouse_pairwise_heatmap(mouse_id: int, matrix_df: pd.DataFrame, output_dir: Path, bin_size_ms: float) -> None:
    plot_heatmap(
        matrix_df=matrix_df,
        title=f"Mouse {mouse_id}: mean neuron-pair correlation ({bin_size_ms:.0f} ms bins)",
        output_path=output_dir / f"mouse_{mouse_id:02d}_pairwise_corr_heatmap.png",
        cbar_label="Mean Pearson r",
    )


def add_pair_labels(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    labeled["pair_label"] = labeled["region_a"] + " vs " + labeled["region_b"]
    return labeled


def plot_group_boxplot(
    combined_df: pd.DataFrame,
    output_dir: Path,
    value_col: str,
    output_name: str,
    title: str,
    ylabel: str,
) -> None:
    plot_df = add_pair_labels(combined_df)
    plot_df = plot_df[np.isfinite(plot_df[value_col])].copy()
    if plot_df.empty:
        return

    order = sorted(plot_df["pair_label"].unique().tolist())

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.boxplot(
        data=plot_df,
        x="pair_label",
        y=value_col,
        order=order,
        color="lightsteelblue",
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="pair_label",
        y=value_col,
        order=order,
        color="black",
        size=4,
        alpha=0.7,
        ax=ax,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Region pair")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_group_mean_heatmap(
    combined_df: pd.DataFrame,
    output_dir: Path,
    value_col: str,
    output_name: str,
    title: str,
    cbar_label: str,
) -> None:
    mean_df = (
        combined_df.groupby(["region_a", "region_b"], as_index=False)[value_col]
        .mean()
        .sort_values(["region_a", "region_b"])
    )
    matrix_df = build_symmetric_matrix(mean_df, value_col=value_col)
    plot_heatmap(
        matrix_df=matrix_df,
        title=title,
        output_path=output_dir / output_name,
        cbar_label=cbar_label,
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.dataset_root / "output" / "region_interactions_100ms"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")

    bin_size_s = args.bin_size_ms / 1000.0
    combined_results: list[pd.DataFrame] = []

    for mouse_id in args.mouse_ids:
        mouse_folder = args.dataset_root / str(mouse_id)
        if not mouse_folder.exists():
            print(f"Skipping mouse {mouse_id}: folder not found at {mouse_folder}")
            continue

        try:
            mouse_df = load_mouse_clusters(mouse_folder)
            interaction_df, population_matrix, pairwise_matrix = pairwise_interactions(
                mouse_df=mouse_df,
                bin_size_s=bin_size_s,
                include_within_region=args.include_within_region,
            )
        except Exception as exc:
            print(f"Skipping mouse {mouse_id}: {exc}")
            continue

        interaction_df.insert(0, "mouse_id", mouse_id)
        save_mouse_results(mouse_id, interaction_df, population_matrix, pairwise_matrix, output_dir)

        if not args.no_plots:
            plot_mouse_population_heatmap(mouse_id, population_matrix, output_dir, args.bin_size_ms)
            plot_mouse_pairwise_heatmap(mouse_id, pairwise_matrix, output_dir, args.bin_size_ms)

        combined_results.append(interaction_df)
        print(
            f"Mouse {mouse_id}: saved {len(interaction_df)} region pairs to {output_dir}"
        )

    if not combined_results:
        raise RuntimeError("No mouse interaction tables were created.")

    combined_df = pd.concat(combined_results, ignore_index=True)
    combined_df.to_csv(output_dir / "all_mice_region_interactions.csv", index=False)

    if not args.no_plots:
        plot_group_mean_heatmap(
            combined_df=combined_df,
            output_dir=output_dir,
            value_col="population_corr",
            output_name="all_mice_population_corr_heatmap.png",
            title="Mean population correlation across mice",
            cbar_label="Pearson r",
        )
        plot_group_boxplot(
            combined_df=combined_df,
            output_dir=output_dir,
            value_col="population_corr",
            output_name="all_mice_population_corr_boxplot.png",
            title="Population correlation across mice",
            ylabel="Pearson r",
        )
        plot_group_mean_heatmap(
            combined_df=combined_df,
            output_dir=output_dir,
            value_col="mean_pairwise_corr",
            output_name="all_mice_pairwise_corr_heatmap.png",
            title="Mean neuron-pair correlation across mice",
            cbar_label="Mean Pearson r",
        )
        plot_group_boxplot(
            combined_df=combined_df,
            output_dir=output_dir,
            value_col="mean_pairwise_corr",
            output_name="all_mice_pairwise_corr_boxplot.png",
            title="Neuron-pair correlation across mice",
            ylabel="Mean Pearson r",
        )

    print(f"Combined results saved to {output_dir / 'all_mice_region_interactions.csv'}")


if __name__ == "__main__":
    main()

