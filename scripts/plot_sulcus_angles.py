"""
Plot sulcus angles per slice from Slicer AxialLabels markups.

For each case folder, reads AxialLabels_<caseID>.mrk.json and computes
the angle between points (1-2) and (2-3) for each slice. Assumes points
are ordered in groups of 3 per slice in ascending slice order.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _compute_angle_deg(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Angle at p2 between vectors p1->p2 and p3->p2."""
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return float("nan")
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _load_points(axial_labels_path: Path) -> List[np.ndarray]:
    with axial_labels_path.open("r", encoding="utf-8") as f:
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


def _group_points(points: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    groups = []
    for i in range(0, len(points) - 2, 3):
        groups.append((points[i], points[i + 1], points[i + 2]))
    return groups


def _plot_case(case_id: str, axial_labels_path: Path, output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required to plot sulcus angles.")

    points = _load_points(axial_labels_path)
    if not points:
        raise RuntimeError(f"No control points found in {axial_labels_path}")

    groups = _group_points(points)
    if not groups:
        raise RuntimeError(f"Not enough points to form angle groups in {axial_labels_path}")

    angles = []
    for p1, p2, p3 in groups:
        angle = _compute_angle_deg(p1, p2, p3)
        angles.append(angle)

    num_slices = len(angles)
    if num_slices == 1:
        x_vals = np.array([0.0])
    else:
        x_vals = np.linspace(0.0, 1.0, num_slices)

    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, angles, marker="o", linewidth=1.5)
    plt.xlabel("Normalized Slice Index (0 to 1)")
    plt.ylabel("Angle (degrees)")
    plt.title(f'Sulcus Angle of patient "{case_id}" for "{num_slices}" Slices')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _infer_case_id(case_dir: Path, axial_labels_path: Path) -> str:
    # Prefer filename suffix if present: AxialLabels_<caseID>.mrk.json
    name = axial_labels_path.stem  # AxialLabels_<caseID>.mrk
    if name.startswith("AxialLabels_"):
        return name[len("AxialLabels_"):].replace(".mrk", "")
    return case_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sulcus angles per case from AxialLabels outputs.")
    parser.add_argument("--input_dir", type=str, default="inference_output",
                        help="Root folder containing per-case output directories.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save plots. Defaults to each case folder.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")

    for case_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        axial_labels_files = list(case_dir.glob("AxialLabels_*.mrk.json"))
        if not axial_labels_files:
            continue
        axial_labels_path = axial_labels_files[0]
        case_id = _infer_case_id(case_dir, axial_labels_path)

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"sulcus_angle_{case_id}.png"
        else:
            output_path = case_dir / f"sulcus_angle_{case_id}.png"

        _plot_case(case_id, axial_labels_path, output_path)
        print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
