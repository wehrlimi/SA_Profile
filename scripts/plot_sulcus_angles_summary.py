"""
Summarize sulcus angles from AxialLabels_*.mrk.json files.
Resamples each case to a fixed number of points (default: 100),
then plots mean ± std across all cases over 0-100% slice position.
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def _compute_angle_deg(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return float("nan")
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _load_points(path: Path) -> List[np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    markups = data.get("markups", [])
    if not markups:
        return []
    control_points = markups[0].get("controlPoints", [])
    points = []
    for cp in control_points:
        pos = cp.get("position")
        if pos and len(pos) == 3:
            points.append(np.array(pos, dtype=float))
    return points


def _angles_from_points(points: List[np.ndarray]) -> List[float]:
    angles = []
    for i in range(0, len(points) - 2, 3):
        p1, p2, p3 = points[i], points[i + 1], points[i + 2]
        angles.append(_compute_angle_deg(p1, p2, p3))
    return angles


def _resample_angles(angles: List[float], num_points: int) -> np.ndarray:
    values = np.array([a for a in angles if np.isfinite(a)], dtype=np.float64)
    if values.size == 0:
        return np.full(num_points, np.nan, dtype=np.float64)
    if values.size == 1:
        return np.full(num_points, values[0], dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, values.size)
    x_new = np.linspace(0.0, 1.0, num_points)
    return np.interp(x_new, x_old, values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mean ± std sulcus angle over normalized slices.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root folder with per-case outputs")
    parser.add_argument("--output", type=str, default="sulcus_angle_summary.png", help="Output plot path")
    parser.add_argument("--num_points", type=int, default=100, help="Number of points for resampling")
    parser.add_argument(
        "--show_individual",
        action="store_true",
        help="Plot each case curve with low alpha",
    )
    parser.add_argument(
        "--individual_alpha",
        type=float,
        default=0.08,
        help="Alpha for individual curves when --show_individual is set",
    )
    parser.add_argument(
        "--angle_min",
        type=float,
        default=None,
        help="Minimum angle to include (degrees). Use with --angle_max.",
    )
    parser.add_argument(
        "--angle_max",
        type=float,
        default=None,
        help="Maximum angle to include (degrees). Use with --angle_min.",
    )
    parser.add_argument(
        "--max_jump",
        type=float,
        default=None,
        help="Exclude cases with angle jumps larger than this threshold (degrees).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")

    per_case: List[np.ndarray] = []
    total_cases = 0
    excluded_cases_jump = 0
    total_angles_raw = 0
    excluded_angles_range = 0
    excluded_angles_jump = 0
    included_angles = []
    files = [
        p for p in input_dir.rglob("AxialLabels_*.mrk.json")
        if "AxialLabels_fitted_" not in p.name
    ]
    for path in files:
        total_cases += 1
        points = _load_points(path)
        if not points:
            continue
        angles = _angles_from_points(points)
        angles = [a for a in angles if np.isfinite(a)]
        total_angles_raw += len(angles)
        if args.angle_min is not None or args.angle_max is not None:
            if args.angle_min is None or args.angle_max is None:
                raise ValueError("--angle_min and --angle_max must be provided together.")
            before = len(angles)
            angles = [a for a in angles if args.angle_min <= a <= args.angle_max]
            excluded_angles_range += before - len(angles)
        if args.max_jump is not None and len(angles) >= 2:
            jumps = np.abs(np.diff(np.array(angles, dtype=np.float64)))
            if np.any(jumps > float(args.max_jump)):
                excluded_cases_jump += 1
                excluded_angles_jump += len(angles)
                continue
        resampled = _resample_angles(angles, args.num_points)
        if np.isfinite(resampled).any():
            per_case.append(resampled)
            included_angles.extend(angles)

    if not per_case:
        raise RuntimeError(f"No valid angles found in {input_dir} (check markups).")

    stacked = np.vstack(per_case)
    mean_angle = np.nanmean(stacked, axis=0)
    std_angle = np.nanstd(stacked, axis=0)
    x_vals = np.linspace(0.0, 100.0, args.num_points)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required to plot the summary.")

    plt.figure(figsize=(8, 4.6))
    if args.show_individual:
        for curve in per_case:
            plt.plot(
                x_vals,
                curve,
                color="tab:blue",
                linewidth=1.0,
                alpha=float(args.individual_alpha),
            )
    plt.plot(x_vals, mean_angle, color="tab:blue", linewidth=1.8, label="Mean angle")
    plt.fill_between(
        x_vals,
        mean_angle - std_angle,
        mean_angle + std_angle,
        color="tab:blue",
        alpha=0.2,
        label="±1 std",
    )
    plt.xlabel("Normalized Slice Position (%)")
    plt.ylabel("Angle (degrees)")
    plt.title("Sulcus angle mean ± std (resampled)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    included_min = float(np.min(included_angles)) if included_angles else float("nan")
    included_max = float(np.max(included_angles)) if included_angles else float("nan")
    included_cases = len(per_case)
    excluded_cases_total = total_cases - included_cases
    angles_included = len(included_angles)
    angles_excluded = total_angles_raw - angles_included
    range_text = "none"
    if args.angle_min is not None and args.angle_max is not None:
        range_text = f"{args.angle_min:.1f}–{args.angle_max:.1f}"
    jump_text = "none" if args.max_jump is None else f">{args.max_jump:.1f}"
    stats_text = (
        f"Cases: {included_cases}/{total_cases} included, {excluded_cases_total} excluded\n"
        f"Angles: {angles_included}/{total_angles_raw} included, {angles_excluded} excluded\n"
        f"Range filter: {range_text} deg, jump filter: {jump_text} deg\n"
        f"Included angle range: {included_min:.1f}–{included_max:.1f} deg"
    )
    plt.gcf().text(0.02, 0.02, stats_text, ha="left", va="bottom", fontsize=9)
    #plt.ylim(0, 180)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()

    print(f"Saved summary plot: {args.output}")
    print(f"Cases: {len(per_case)}, points per case: {args.num_points}")


if __name__ == "__main__":
    main()
