from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
        default=Path("./brainhack-dataset"),
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
    parser.add_argument(
        "--include-within-region",
        action="store_true",
        help="Also compute within-region interactions in addition to across-region pairs.",
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


def build_population_matrix(result_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    matrix_df = build_population_matrix(result_df, value_col="population_corr")
    return result_df, matrix_df


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

    pop_trace_a = binned[idx_a].sum(axis=0)
    pop_trace_b = binned[idx_b].sum(axis=0)
    population_corr = correlation_or_nan(pop_trace_a, pop_trace_b)

    abs_correlations = np.abs(correlations) if correlations else []

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
            "mean_abs_pairwise_corr": float(np.mean(abs_correlations)) if correlations else np.nan,
            "median_abs_pairwise_corr": float(np.median(abs_correlations)) if correlations else np.nan,
            "population_corr": population_corr,
        }
    ]


def save_mouse_results(
    mouse_id: int,
    interaction_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    interaction_path = output_dir / f"mouse_{mouse_id:02d}_region_interactions.csv"
    matrix_path = output_dir / f"mouse_{mouse_id:02d}_population_corr_matrix.csv"

    interaction_df.to_csv(interaction_path, index=False)
    matrix_df.to_csv(matrix_path)


def plot_heatmap(
    matrix_df: pd.DataFrame,
    title: str,
    output_path: Path,
    cmap: str = "coolwarm",
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix_df,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Brain region")
    ax.set_ylabel("Brain region")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mouse_heatmap(mouse_id: int, matrix_df: pd.DataFrame, output_dir: Path) -> None:
    plot_heatmap(
        matrix_df=matrix_df,
        title=f"Mouse {mouse_id}: population correlation (100 ms bins)",
        output_path=output_dir / f"mouse_{mouse_id:02d}_population_corr_heatmap.png",
    )


def add_pair_labels(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    labeled["pair_label"] = labeled["region_a"] + " vs " + labeled["region_b"]
    return labeled


def plot_group_pair_summary(combined_df: pd.DataFrame, output_dir: Path) -> None:
    summary_df = add_pair_labels(combined_df)
    order = sorted(summary_df["pair_label"].unique().tolist())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(
        data=summary_df,
        x="pair_label",
        y="population_corr",
        order=order,
        color="lightsteelblue",
        ax=ax,
    )
    sns.stripplot(
        data=summary_df,
        x="pair_label",
        y="population_corr",
        order=order,
        color="black",
        size=4,
        alpha=0.7,
        ax=ax,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Population correlation across mice")
    ax.set_xlabel("Region pair")
    ax.set_ylabel("Pearson r")
    fig.tight_layout()
    fig.savefig(output_dir / "all_mice_population_corr_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_group_mean_heatmap(combined_df: pd.DataFrame, output_dir: Path) -> None:
    mean_df = (
        combined_df.groupby(["region_a", "region_b"], as_index=False)["population_corr"]
        .mean()
        .sort_values(["region_a", "region_b"])
    )
    matrix_df = build_population_matrix(mean_df, value_col="population_corr")
    plot_heatmap(
        matrix_df=matrix_df,
        title="Mean population correlation across mice",
        output_path=output_dir / "all_mice_population_corr_heatmap.png",
    )


def plot_within_area_abs_summary(combined_df: pd.DataFrame, output_dir: Path) -> None:
    within_df = combined_df[combined_df["region_a"] == combined_df["region_b"]].copy()
    within_df["brain_area"] = within_df["region_a"]

    area_order = (
        within_df.groupby("brain_area")["mean_abs_pairwise_corr"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=within_df,
        x="brain_area",
        y="mean_abs_pairwise_corr",
        order=area_order,
        color="lightsteelblue",
        ax=ax,
    )
    sns.stripplot(
        data=within_df,
        x="brain_area",
        y="mean_abs_pairwise_corr",
        order=area_order,
        color="black",
        size=4,
        alpha=0.7,
        ax=ax,
    )

    ax.set_title("Within-area interaction strength across mice (100 ms bins)")
    ax.set_xlabel("Brain area")
    ax.set_ylabel("Mean |Pearson r| across neuron pairs")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(
        output_dir / "all_mice_within_area_mean_abs_pairwise_corr.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output_dir = args.output_dir or dataset_root / "output" / "region_interactions_100ms"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")

    bin_size_s = args.bin_size_ms / 1000.0
    combined_results: list[pd.DataFrame] = []

    for mouse_id in args.mouse_ids:
        mouse_folder = dataset_root / str(mouse_id)
        if not mouse_folder.exists():
            print(f"Skipping mouse {mouse_id}: folder not found at {mouse_folder}")
            continue

        try:
            mouse_df = load_mouse_clusters(mouse_folder)
            interaction_df, matrix_df = pairwise_interactions(
                mouse_df=mouse_df,
                bin_size_s=bin_size_s,
                include_within_region=True,
            )
        except Exception as exc:
            print(f"Skipping mouse {mouse_id}: {exc}")
            continue

        interaction_df.insert(0, "mouse_id", mouse_id)
        save_mouse_results(mouse_id, interaction_df, matrix_df, output_dir)

        if not args.no_plots:
            plot_mouse_heatmap(mouse_id, matrix_df, output_dir)

        combined_results.append(interaction_df)
        print(
            f"Mouse {mouse_id}: saved {len(interaction_df)} region pairs to "
            f"{output_dir}"
        )

    if not combined_results:
        raise RuntimeError("No mouse interaction tables were created.")

    combined_df = pd.concat(combined_results, ignore_index=True)
    within_df = combined_df[
        combined_df["region_a"] == combined_df["region_b"]
    ].copy()

    within_df["brain_area"] = within_df["region_a"]
    area_ranking_df = (
        within_df.groupby("brain_area", as_index=False)
        .agg(
            mean_abs_pairwise_corr=("mean_abs_pairwise_corr", "mean"),
            median_abs_pairwise_corr=("median_abs_pairwise_corr", "mean"),
            mean_signed_pairwise_corr=("mean_pairwise_corr", "mean"),
            mice_count=("mouse_id", "nunique"),
        )
        .sort_values("mean_abs_pairwise_corr", ascending=False)
        .reset_index(drop=True)
    )
    top_area = area_ranking_df.iloc[0]
    print(
        f"Strongest brain area at {int(args.bin_size_ms)} ms "
        f"(by mean absolute pairwise correlation): "
        f"{top_area['brain_area']} "
        f"[score={top_area['mean_abs_pairwise_corr']:.4f}]"
    )

    area_ranking_df.to_csv(output_dir / "brain_area_abs_interaction_ranking.csv", index=False)
    combined_df.to_csv(output_dir / "all_mice_region_interactions.csv", index=False)

    if not args.no_plots:
        plot_within_area_abs_summary(combined_df, output_dir)
        plot_group_mean_heatmap(combined_df, output_dir)
        plot_group_pair_summary(combined_df, output_dir)

    print(f"Combined results saved to {output_dir / 'all_mice_region_interactions.csv'}")


if __name__ == "__main__":
    main()
