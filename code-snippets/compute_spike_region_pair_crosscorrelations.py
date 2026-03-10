from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# This script computes spike-train cross-correlations between neurons from pairs of brain regions.
#
# Workflow:
# 1. Load spikes.npy + clusters.npy + brain_area.npy for each mouse.
# 2. Regroup spikes into one spike train per neuron / cluster.
# 3. Bin every neuron's spikes using a tunable bin width (default 20 ms).
# 4. For each brain-region pair (A, B), compute the lagged cross-correlation for every
#    neuron pair with source neuron in region A and target neuron in region B.
# 5. Save the full lag curve for every neuron pair, plus a region-pair average curve.
#
# Convention for lag sign:
# - We compute corr( neuron_A[t], neuron_B[t + lag] ).
# - Positive lag means neuron_B (and therefore region B) tends to fire after neuron_A.
# - Negative lag means neuron_B tends to fire before neuron_A.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute neuron-pair spike cross-correlations between brain-region pairs "
            "with tunable spike-count bin width."
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
            "Optional output directory. By default, results go to "
            "<dataset-root>/output/spike_crosscorrelations_<bin>ms."
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
        default=40.0,
        help="Spike bin width in milliseconds.",
    )
    parser.add_argument(
        "--max-lag-ms",
        type=float,
        default=1000.0,
        help="Maximum lag on either side of zero, in milliseconds.",
    )
    parser.add_argument(
        "--min-spikes-per-neuron",
        type=int,
        default=20,
        help="Minimum spikes required for a neuron to be included.",
    )
    parser.add_argument(
        "--include-within-region",
        action="store_true",
        help="Also compute neuron-pair cross-correlations within each region.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving summary plots and only export CSV files.",
    )
    return parser.parse_args()


def format_region_label(region: object) -> str:
    if isinstance(region, (bytes, np.bytes_)):
        return region.decode("utf-8")
    return str(region)


def default_output_dir(dataset_root: Path, bin_size_ms: float) -> Path:
    bin_label = f"{int(round(bin_size_ms)):03d}ms"
    return dataset_root / "output" / f"spike_crosscorrelations_{bin_label}"


def load_mouse_neurons(mouse_folder: Path) -> pd.DataFrame:
    spikes = np.load(mouse_folder / "spikes.npy")
    clusters = np.load(mouse_folder / "clusters.npy")
    brain_area_info = np.load(mouse_folder / "brain_area.npy", allow_pickle=True).item()

    region_map = dict(zip(brain_area_info["cluster_id"], brain_area_info["brain_area"]))

    grouped_spikes: dict[int, list[float]] = {}
    for cluster_id, spike_time in zip(clusters, spikes):
        grouped_spikes.setdefault(int(cluster_id), []).append(float(spike_time))

    rows: list[dict[str, object]] = []
    for cluster_id, spike_times in grouped_spikes.items():
        region = region_map.get(cluster_id)
        if region is None:
            continue
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "brain_region": format_region_label(region),
                "spike_times": np.sort(np.asarray(spike_times, dtype=float)),
                "n_spikes": int(len(spike_times)),
            }
        )

    if not rows:
        raise ValueError(f"No neurons with region assignments found in {mouse_folder}")

    return pd.DataFrame(rows).sort_values(["brain_region", "cluster_id"]).reset_index(drop=True)


def build_time_bins(neuron_df: pd.DataFrame, bin_size_ms: float) -> tuple[np.ndarray, float, float]:
    bin_size_s = bin_size_ms / 1000.0
    all_spikes = np.concatenate(neuron_df["spike_times"].to_numpy())
    recording_start = 0.0 if np.nanmin(all_spikes) >= 0 else float(np.nanmin(all_spikes))
    recording_end = float(np.nanmax(all_spikes)) + bin_size_s
    bin_edges = np.arange(recording_start, recording_end + bin_size_s, bin_size_s)
    return bin_edges, recording_start, bin_size_s
    

def bin_neuron_spikes(neuron_df: pd.DataFrame, bin_edges: np.ndarray, recording_start: float) -> pd.DataFrame:
    n_bins = bin_edges.size - 1
    binned_counts: list[np.ndarray] = []

    for row in neuron_df.itertuples(index=False):
        bin_ids = np.floor((row.spike_times - recording_start) / (bin_edges[1] - bin_edges[0])).astype(int)
        valid = (bin_ids >= 0) & (bin_ids < n_bins)
        counts = np.bincount(bin_ids[valid], minlength=n_bins).astype(float)
        binned_counts.append(counts)

    binned_df = neuron_df.copy()
    binned_df["binned_counts"] = binned_counts
    return binned_df


def compute_lagged_cross_correlation(
    source_counts: np.ndarray,
    target_counts: np.ndarray,
    max_lag_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    # For each lag k, we correlate the overlapping parts of:
    #   source[t] with target[t + k]
    #
    # This returns a Pearson correlation coefficient for each lag, which makes curves
    # comparable across neuron pairs despite different firing rates.
    lags = np.arange(-max_lag_bins, max_lag_bins + 1, dtype=int)
    corr_values = np.full(lags.shape, np.nan, dtype=float)

    for idx, lag in enumerate(lags):
        if lag < 0:
            source_overlap = source_counts[-lag:]
            target_overlap = target_counts[: source_counts.size + lag]
        elif lag > 0:
            source_overlap = source_counts[: source_counts.size - lag]
            target_overlap = target_counts[lag:]
        else:
            source_overlap = source_counts
            target_overlap = target_counts

        if source_overlap.size < 2:
            continue
        source_std = float(np.std(source_overlap))
        target_std = float(np.std(target_overlap))
        if source_std == 0 or target_std == 0:
            continue

        source_centered = source_overlap - np.mean(source_overlap)
        target_centered = target_overlap - np.mean(target_overlap)
        corr_values[idx] = float(np.mean(source_centered * target_centered) / (source_std * target_std))

    return lags, corr_values


def compute_mouse_crosscorrelations(
    neuron_df: pd.DataFrame,
    bin_size_ms: float,
    max_lag_ms: float,
    min_spikes_per_neuron: int,
    include_within_region: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    bin_edges, recording_start, bin_size_s = build_time_bins(neuron_df, bin_size_ms)
    binned_df = bin_neuron_spikes(neuron_df, bin_edges, recording_start)
    max_lag_bins = int(round(max_lag_ms / bin_size_ms))

    if max_lag_bins < 0:
        raise ValueError("max-lag-ms must be non-negative.")

    regions = sorted(binned_df["brain_region"].dropna().unique().tolist())
    region_pairs = list(combinations(regions, 2))
    if include_within_region:
        region_pairs.extend((region, region) for region in regions)

    pair_rows: list[dict[str, object]] = []
    mean_rows: list[dict[str, object]] = []
    lag_axis_ms: np.ndarray | None = None

    for region_a, region_b in region_pairs:
        region_a_df = binned_df[binned_df["brain_region"] == region_a]
        region_b_df = binned_df[binned_df["brain_region"] == region_b]

        region_a_valid = region_a_df[region_a_df["n_spikes"] >= min_spikes_per_neuron]
        region_b_valid = region_b_df[region_b_df["n_spikes"] >= min_spikes_per_neuron]

        pair_curves: list[np.ndarray] = []
        n_pairs = 0

        if region_a == region_b:
            # Within-region mode uses unique neuron pairs and excludes autocorrelations.
            tuples = list(region_a_valid.itertuples(index=False))
            for i in range(len(tuples)):
                for j in range(i + 1, len(tuples)):
                    source_row = tuples[i]
                    target_row = tuples[j]
                    lag_bins, corr_values = compute_lagged_cross_correlation(
                        source_counts=source_row.binned_counts,
                        target_counts=target_row.binned_counts,
                        max_lag_bins=max_lag_bins,
                    )
                    lag_axis_ms = lag_bins * bin_size_ms
                    n_pairs += 1
                    pair_curves.append(corr_values)
                    for lag_ms, corr_value in zip(lag_axis_ms, corr_values):
                        pair_rows.append(
                            {
                                "region_a": region_a,
                                "region_b": region_b,
                                "cluster_a": int(source_row.cluster_id),
                                "cluster_b": int(target_row.cluster_id),
                                "lag_ms": float(lag_ms),
                                "cross_correlation": float(corr_value) if np.isfinite(corr_value) else np.nan,
                            }
                        )
        else:
            for source_row in region_a_valid.itertuples(index=False):
                for target_row in region_b_valid.itertuples(index=False):
                    lag_bins, corr_values = compute_lagged_cross_correlation(
                        source_counts=source_row.binned_counts,
                        target_counts=target_row.binned_counts,
                        max_lag_bins=max_lag_bins,
                    )
                    lag_axis_ms = lag_bins * bin_size_ms
                    n_pairs += 1
                    pair_curves.append(corr_values)
                    for lag_ms, corr_value in zip(lag_axis_ms, corr_values):
                        pair_rows.append(
                            {
                                "region_a": region_a,
                                "region_b": region_b,
                                "cluster_a": int(source_row.cluster_id),
                                "cluster_b": int(target_row.cluster_id),
                                "lag_ms": float(lag_ms),
                                "cross_correlation": float(corr_value) if np.isfinite(corr_value) else np.nan,
                            }
                        )

        if lag_axis_ms is None:
            lag_axis_ms = np.arange(-max_lag_bins, max_lag_bins + 1, dtype=float) * bin_size_ms

        if pair_curves:
            curve_stack = np.vstack(pair_curves)
            mean_curve = np.nanmean(curve_stack, axis=0)
            std_curve = np.nanstd(curve_stack, axis=0)
        else:
            mean_curve = np.full(lag_axis_ms.shape, np.nan, dtype=float)
            std_curve = np.full(lag_axis_ms.shape, np.nan, dtype=float)

        for lag_ms, mean_corr, std_corr in zip(lag_axis_ms, mean_curve, std_curve):
            mean_rows.append(
                {
                    "region_a": region_a,
                    "region_b": region_b,
                    "lag_ms": float(lag_ms),
                    "mean_cross_correlation": float(mean_corr) if np.isfinite(mean_corr) else np.nan,
                    "std_cross_correlation": float(std_corr) if np.isfinite(std_corr) else np.nan,
                    "n_neuron_pairs": int(n_pairs),
                    "n_region_a_neurons": int(len(region_a_df)),
                    "n_region_b_neurons": int(len(region_b_df)),
                    "n_region_a_neurons_used": int(len(region_a_valid)),
                    "n_region_b_neurons_used": int(len(region_b_valid)),
                }
            )

    pair_df = pd.DataFrame(pair_rows)
    mean_df = pd.DataFrame(mean_rows)
    return pair_df, mean_df, lag_axis_ms


def plot_mouse_mean_curves(mouse_id: int, mean_df: pd.DataFrame, output_dir: Path) -> None:
    region_pairs = (
        mean_df[["region_a", "region_b"]]
        .drop_duplicates()
        .sort_values(["region_a", "region_b"])
        .itertuples(index=False, name=None)
    )
    region_pairs = list(region_pairs)
    n_pairs = len(region_pairs)
    n_cols = 3
    n_rows = int(np.ceil(max(n_pairs, 1) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.8, n_rows * 3.8), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()

    for idx, (region_a, region_b) in enumerate(region_pairs):
        ax = axes_arr[idx]
        pair_df = mean_df[(mean_df["region_a"] == region_a) & (mean_df["region_b"] == region_b)].sort_values("lag_ms")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.plot(pair_df["lag_ms"], pair_df["mean_cross_correlation"], color="navy", linewidth=1.8)
        ax.fill_between(
            pair_df["lag_ms"],
            pair_df["mean_cross_correlation"] - pair_df["std_cross_correlation"],
            pair_df["mean_cross_correlation"] + pair_df["std_cross_correlation"],
            color="skyblue",
            alpha=0.3,
        )
        n_pairs_used = int(pair_df["n_neuron_pairs"].iloc[0]) if not pair_df.empty else 0
        ax.set_title(f"{region_a} vs {region_b}\npairs={n_pairs_used}")
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Cross-correlation")

    for idx in range(n_pairs, axes_arr.size):
        axes_arr[idx].axis("off")

    fig.suptitle(
        f"Mouse {mouse_id}: mean neuron-pair spike cross-correlations\n"
        "Positive lag means region_b spikes occur after region_a spikes.",
        fontsize=14,
    )
    fig.savefig(output_dir / f"mouse_{mouse_id:02d}_mean_crosscorrelations.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_mice_mean_curves(all_mean_df: pd.DataFrame, output_dir: Path) -> None:
    group_df = (
        all_mean_df.groupby(["region_a", "region_b", "lag_ms"], as_index=False)["mean_cross_correlation"]
        .mean()
        .sort_values(["region_a", "region_b", "lag_ms"])
    )

    region_pairs = (
        group_df[["region_a", "region_b"]]
        .drop_duplicates()
        .sort_values(["region_a", "region_b"])
        .itertuples(index=False, name=None)
    )
    region_pairs = list(region_pairs)
    n_pairs = len(region_pairs)
    n_cols = 3
    n_rows = int(np.ceil(max(n_pairs, 1) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.8, n_rows * 3.8), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()

    for idx, (region_a, region_b) in enumerate(region_pairs):
        ax = axes_arr[idx]
        pair_df = group_df[(group_df["region_a"] == region_a) & (group_df["region_b"] == region_b)].sort_values("lag_ms")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.plot(pair_df["lag_ms"], pair_df["mean_cross_correlation"], color="darkgreen", linewidth=1.8)
        ax.set_title(f"{region_a} vs {region_b}")
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Mean cross-correlation")

    for idx in range(n_pairs, axes_arr.size):
        axes_arr[idx].axis("off")

    fig.suptitle("Mean neuron-pair spike cross-correlations across mice", fontsize=16)
    fig.savefig(output_dir / "all_mice_mean_crosscorrelations.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_lag_ms < 0:
        raise ValueError("max-lag-ms must be non-negative.")

    output_dir = args.output_dir or default_output_dir(args.dataset_root, args.bin_size_ms)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")

    all_pair_tables: list[pd.DataFrame] = []
    all_mean_tables: list[pd.DataFrame] = []

    for mouse_id in args.mouse_ids:
        mouse_folder = args.dataset_root / str(mouse_id)
        if not mouse_folder.exists():
            print(f"Skipping mouse {mouse_id}: folder not found at {mouse_folder}")
            continue

        try:
            neuron_df = load_mouse_neurons(mouse_folder)
            pair_df, mean_df, _ = compute_mouse_crosscorrelations(
                neuron_df=neuron_df,
                bin_size_ms=args.bin_size_ms,
                max_lag_ms=args.max_lag_ms,
                min_spikes_per_neuron=args.min_spikes_per_neuron,
                include_within_region=args.include_within_region,
            )
        except Exception as exc:
            print(f"Skipping mouse {mouse_id}: {exc}")
            continue

        pair_df.insert(0, "mouse_id", int(mouse_id))
        mean_df.insert(0, "mouse_id", int(mouse_id))
        pair_df.to_csv(output_dir / f"mouse_{mouse_id:02d}_pairwise_crosscorrelations.csv", index=False)
        mean_df.to_csv(output_dir / f"mouse_{mouse_id:02d}_region_pair_mean_crosscorrelations.csv", index=False)

        if not args.no_plots and not mean_df.empty:
            plot_mouse_mean_curves(mouse_id, mean_df, output_dir)

        all_pair_tables.append(pair_df)
        all_mean_tables.append(mean_df)
        print(
            f"Mouse {mouse_id}: saved {len(pair_df)} lag rows of pairwise cross-correlations to {output_dir}"
        )

    if not all_pair_tables:
        raise RuntimeError("No cross-correlation tables were created.")

    all_pair_df = pd.concat(all_pair_tables, ignore_index=True)
    all_mean_df = pd.concat(all_mean_tables, ignore_index=True)
    all_pair_df.to_csv(output_dir / "all_mice_pairwise_crosscorrelations.csv", index=False)
    all_mean_df.to_csv(output_dir / "all_mice_region_pair_mean_crosscorrelations.csv", index=False)

    if not args.no_plots and not all_mean_df.empty:
        plot_all_mice_mean_curves(all_mean_df, output_dir)

    print(f"Combined cross-correlations saved to {output_dir / 'all_mice_pairwise_crosscorrelations.csv'}")


if __name__ == "__main__":
    main()
