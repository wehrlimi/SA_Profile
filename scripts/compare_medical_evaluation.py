"""
Compare manual sulcus angle measurements (Medical_Evaluation) against
automated inference outputs (inference_output_11b_w_plots / 11c_w_plots).

Produces two plots:
  1. Boxplot of angular deviation (inference - manual) for fastMRI (11b) and UKBB (11c).
  2. Continuous mean±std sulcus-angle curves from 11b and 11c with
     manually labelled patients overlaid as scatter points.

Usage:
  python scripts/compare_medical_evaluation.py --output_prefix comparison_medical
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
try:
    matplotlib.use("Agg")
except Exception:
    pass
import matplotlib.pyplot as plt


# ─── helpers ────────────────────────────────────────────────────────────────

def compute_angle_deg_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Angle at *p2* between rays p2→p1 and p2→p3  (2-D, degrees)."""
    v1 = p1 - p2
    v2 = p3 - p2
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def load_trochlea_points(mrk_path: Path) -> Optional[List[np.ndarray]]:
    """Return the 3 TrochleaPoints as 3-D arrays (LPS, mm), or None."""
    with open(mrk_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cps = data.get("markups", [{}])[0].get("controlPoints", [])
    if len(cps) < 3:
        return None
    # Take the last 3 control points (in case earlier ones are from previous attempts)
    pts = [np.array(cp["position"], dtype=np.float64) for cp in cps[-3:]]
    return pts


def load_case_metadata(meta_path: Path) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_axial_labels_z_per_slice(mrk_path: Path) -> List[float]:
    """Return a list of z-coordinates (S in LPS), one per axial slice.

    The AxialLabels markup stores 3 points per slice. We average the
    z-component of each group of 3.
    """
    with open(mrk_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cps = data.get("markups", [{}])[0].get("controlPoints", [])
    z_per_slice = []
    for i in range(0, len(cps) - 2, 3):
        z_avg = np.mean([cps[i + j]["position"][2] for j in range(3)])
        z_per_slice.append(z_avg)
    return z_per_slice


def load_sulcus_angles(json_path: Path) -> List[float]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("angles", [])


def _resample_angles(angles: List[float], num_points: int,
                     min_angle: Optional[float] = None,
                     max_jump: Optional[float] = None) -> Optional[np.ndarray]:
    """Resample raw angle list to *num_points* evenly-spaced values.

    Returns None if the case should be excluded (jump filter).
    """
    vals = np.array(angles, dtype=np.float64)
    if min_angle is not None:
        vals[vals < min_angle] = np.nan
    if max_jump is not None and len(vals) > 1:
        if np.any(np.abs(np.diff(vals[np.isfinite(vals)])) > max_jump):
            return None
    valid = vals[np.isfinite(vals)]
    if valid.size < 2:
        return None
    x_old = np.linspace(0.0, 1.0, valid.size)
    x_new = np.linspace(0.0, 1.0, num_points)
    return np.interp(x_new, x_old, valid)


# ─── matching helpers ───────────────────────────────────────────────────────

def _extract_fastmri_key(name: str) -> Optional[Tuple[str, str]]:
    """Extract (patient_id, study_id) from a fastMRI folder name.

    Works for both:
      Medical eval:  FB_105209____FB_2950978745_study_39e3351f_MR2_...
      Inference:     FB_105209____FB,2950978745_study_39e3351f_res_...
    """
    m = re.search(r"(FB_\d+)", name)
    s = re.search(r"study_([a-f0-9]+)", name)
    if m and s:
        return m.group(1), s.group(1)
    return None


def _extract_ukbb_key(name: str) -> Optional[Tuple[str, str]]:
    """Extract (patient_hex, session_AA) from a UKBB folder name.

    Medical eval:  00000BC8_ANONMHP3VI1FO_AA2E75AD_AA5C3F26_...
      → key = ("00000BC8", "AA5C3F26")  (the AA that also appears in inference)
    Inference:     00000BC8_AA5C3F26_res_256_...
      → key = ("00000BC8", "AA5C3F26")
    """
    # Inference style: HEXID_AAHEXID_res_...
    m = re.match(r"([0-9A-F]{8})_(AA[0-9A-F]+)_res_", name, re.I)
    if m:
        return m.group(1).upper(), m.group(2).upper()
    # Medical eval style – several AA segments; we need to figure out which AA
    # is the session one. It's the one that shows up in the inference folder.
    # Strategy: extract patient id + all AA segments, try to match later.
    m2 = re.match(r"([0-9A-F]{8})_", name, re.I)
    if m2:
        patient_id = m2.group(1).upper()
        aa_parts = re.findall(r"(AA[0-9A-F]+)", name, re.I)
        return patient_id, tuple(a.upper() for a in aa_parts)  # type: ignore[return-value]
    return None


def build_inference_index(inf_dir: Path, dataset: str) -> Dict[str, Path]:
    """Map canonical key → case folder inside an inference output directory."""
    index: Dict[str, Path] = {}
    for p in sorted(inf_dir.iterdir()):
        if not p.is_dir():
            continue
        if dataset == "fastmri":
            key = _extract_fastmri_key(p.name)
            if key:
                index[key] = p
        else:
            key = _extract_ukbb_key(p.name)
            if key:
                index[key] = p
    return index


# ─── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Medical Evaluation manual angles to inference outputs"
    )
    parser.add_argument("--medical_eval_dir", type=str,
                        default=r"Medical_Evaluation\Knee_medical_evaluation_out\Knee out",
                        help="Folder with per-case medical evaluation subfolders")
    parser.add_argument("--inference_11b", type=str,
                        default="inference_output_11b_w_plots",
                        help="fastMRI inference output directory")
    parser.add_argument("--inference_11c", type=str,
                        default="inference_output_11c_w_plots",
                        help="UKBB inference output directory")
    parser.add_argument("--output_prefix", type=str,
                        default="comparison_medical",
                        help="Prefix for output PNG files")
    parser.add_argument("--num_points", type=int, default=100,
                        help="Resampling points for continuous curves")
    parser.add_argument("--max_jump", type=float, default=10,
                        help="Exclude inference cases with angle jumps > this")
    parser.add_argument("--min_angle", type=float, default=75,
                        help="Exclude angle values below this (degrees)")
    parser.add_argument("--whitelist_fastmri", type=str, default=None,
                        help="Text file with allowed fastMRI inference dir names")
    parser.add_argument("--whitelist_ukbb", type=str, default=None,
                        help="Text file with allowed UKBB inference dir names")
    args = parser.parse_args()

    med_dir = Path(args.medical_eval_dir)
    inf_11b = Path(args.inference_11b)
    inf_11c = Path(args.inference_11c)
    num_pts = args.num_points

    if not med_dir.is_dir():
        raise FileNotFoundError(f"Medical evaluation dir not found: {med_dir}")
    if not inf_11b.is_dir():
        raise FileNotFoundError(f"Inference 11b dir not found: {inf_11b}")
    if not inf_11c.is_dir():
        raise FileNotFoundError(f"Inference 11c dir not found: {inf_11c}")

    # Load whitelists (if provided)
    def _load_whitelist(path_str):
        if path_str is None:
            return None
        wl = set()
        with open(path_str, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    wl.add(line)
        return wl

    wl_fastmri = _load_whitelist(args.whitelist_fastmri)
    wl_ukbb    = _load_whitelist(args.whitelist_ukbb)

    # Build inference folder indexes
    idx_11b = build_inference_index(inf_11b, "fastmri")
    idx_11c = build_inference_index(inf_11c, "ukbb")

    # Apply whitelists to indexes
    if wl_fastmri is not None:
        idx_11b = {k: v for k, v in idx_11b.items() if v.name in wl_fastmri}
    if wl_ukbb is not None:
        idx_11c = {k: v for k, v in idx_11c.items() if v.name in wl_ukbb}

    print(f"Inference index 11b: {len(idx_11b)} cases"
          + (f" (whitelist: {len(wl_fastmri)})" if wl_fastmri else ""))
    print(f"Inference index 11c: {len(idx_11c)} cases"
          + (f" (whitelist: {len(wl_ukbb)})" if wl_ukbb else ""))

    # ── per-case matching ──────────────────────────────────────────────────
    # Collect: dataset, manual_angle, inference_angle, norm_position
    records: List[dict] = []
    matched_ukbb_dirs: set = set()

    for case_dir in sorted(med_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        trochlea_path = case_dir / "TrochleaPoints.mrk.json"
        meta_path = case_dir / "case_metadata.json"
        if not trochlea_path.exists() or not meta_path.exists():
            continue

        # Determine dataset
        folder_name = case_dir.name
        is_fastmri = folder_name.startswith("FB_")

        # Load manual data
        pts = load_trochlea_points(trochlea_path)
        if pts is None or len(pts) < 3:
            print(f"  SKIP (no points): {folder_name}")
            continue
        # Manual sulcus angle in the axial plane (use L, P = first 2 coords)
        manual_angle = compute_angle_deg_2d(pts[0][:2], pts[1][:2], pts[2][:2])
        if not np.isfinite(manual_angle):
            print(f"  SKIP (NaN angle): {folder_name}")
            continue
        # World z of the measurement (S in LPS = 3rd coord)
        manual_z = float(np.mean([p[2] for p in pts]))

        # Match to inference
        if is_fastmri:
            key = _extract_fastmri_key(folder_name)
            inf_case_dir = idx_11b.get(key) if key else None
            dataset_label = "fastMRI (11b)"
        else:
            raw_key = _extract_ukbb_key(folder_name)
            inf_case_dir = None
            if raw_key is not None:
                patient_id, aa_tuple = raw_key[0], raw_key[1]
                # aa_tuple is a tuple of all AA segments from the medical eval folder
                # try each one against the inference index
                if isinstance(aa_tuple, tuple):
                    for aa in aa_tuple:
                        cand_key = (patient_id, aa)
                        if cand_key in idx_11c:
                            inf_case_dir = idx_11c[cand_key]
                            break
                else:
                    cand_key = (patient_id, aa_tuple)
                    inf_case_dir = idx_11c.get(cand_key)
            dataset_label = "UKBB (11c)"

        if inf_case_dir is None:
            print(f"  NO MATCH: {folder_name}")
            continue

        # Track matched UKBB inference dirs for Plot 2 curve selection
        if not is_fastmri:
            matched_ukbb_dirs.add(inf_case_dir)

        # Load inference sulcus angles
        angle_files = list(inf_case_dir.glob("sulcus_angles_*.json"))
        if not angle_files:
            print(f"  NO ANGLES JSON: {inf_case_dir.name}")
            continue
        inf_angles = load_sulcus_angles(angle_files[0])
        if not inf_angles:
            print(f"  EMPTY ANGLES: {inf_case_dir.name}")
            continue
        num_slices_inf = len(inf_angles)

        # Load fitted AxialLabels to get z-coordinates per slice
        fitted_labels = list(inf_case_dir.glob("AxialLabels_fitted_*.mrk.json"))
        if not fitted_labels:
            fitted_labels = list(inf_case_dir.glob("AxialLabels_*.mrk.json"))
        if not fitted_labels:
            print(f"  NO AXIAL LABELS: {inf_case_dir.name}")
            continue

        z_per_slice = load_axial_labels_z_per_slice(fitted_labels[0])
        if len(z_per_slice) != num_slices_inf:
            print(f"  MISMATCH slices ({len(z_per_slice)} vs {num_slices_inf}): "
                  f"{inf_case_dir.name}")
            # Try raw labels as backup
            raw_labels = list(inf_case_dir.glob("AxialLabels_*.mrk.json"))
            for rl in raw_labels:
                if "fitted" not in rl.name:
                    z_per_slice2 = load_axial_labels_z_per_slice(rl)
                    if len(z_per_slice2) == num_slices_inf:
                        z_per_slice = z_per_slice2
                        break
        if len(z_per_slice) != num_slices_inf:
            print(f"  STILL MISMATCH, skipping: {inf_case_dir.name}")
            continue

        # Find closest inference slice to the manual z-coordinate
        z_arr = np.array(z_per_slice)
        closest_idx = int(np.argmin(np.abs(z_arr - manual_z)))
        inf_angle_at_slice = inf_angles[closest_idx]

        # Normalized position (0-100%)
        if num_slices_inf > 1:
            norm_pos = closest_idx / (num_slices_inf - 1) * 100.0
        else:
            norm_pos = 50.0

        deviation = inf_angle_at_slice - manual_angle

        records.append({
            "dataset": dataset_label,
            "manual_angle": manual_angle,
            "inference_angle": inf_angle_at_slice,
            "deviation": deviation,
            "norm_position": norm_pos,
            "case_med": folder_name,
            "case_inf": inf_case_dir.name,
            "manual_z": manual_z,
            "matched_z": float(z_arr[closest_idx]),
            "z_distance_mm": float(abs(manual_z - z_arr[closest_idx])),
            "inf_angles_full": inf_angles,       # full curve for per-case plots
            "num_slices_inf": num_slices_inf,
        })
        print(f"  OK  {dataset_label:16s}  manual={manual_angle:6.1f}  "
              f"inf={inf_angle_at_slice:6.1f}  dev={deviation:+6.1f}  "
              f"pos={norm_pos:5.1f}%  dz={abs(manual_z - z_arr[closest_idx]):.2f}mm  "
              f"{folder_name[:50]}")

    if not records:
        print("\nNo matched cases found – nothing to plot.")
        return

    # Split by dataset
    fastmri_recs = [r for r in records if "fastMRI" in r["dataset"]]
    ukbb_recs    = [r for r in records if "UKBB" in r["dataset"]]
    print(f"\nMatched cases (with manual):  fastMRI = {len(fastmri_recs)},  UKBB = {len(ukbb_recs)}")

    # ── Collect inference-only cases (in whitelist but no manual measurement) ──
    matched_inf_dirs_fm  = {r["case_inf"] for r in fastmri_recs}
    matched_inf_dirs_ukbb = {r["case_inf"] for r in ukbb_recs}

    extra_inf_fm: List[dict] = []   # inference-only fastMRI
    extra_inf_ukbb: List[dict] = []  # inference-only UKBB
    for key, inf_case_dir in idx_11b.items():
        if inf_case_dir.name not in matched_inf_dirs_fm:
            angle_files = list(inf_case_dir.glob("sulcus_angles_*.json"))
            if angle_files:
                inf_angles = load_sulcus_angles(angle_files[0])
                if inf_angles:
                    extra_inf_fm.append({
                        "case_inf": inf_case_dir.name,
                        "inf_angles_full": inf_angles,
                        "num_slices_inf": len(inf_angles),
                    })
    for key, inf_case_dir in idx_11c.items():
        if inf_case_dir.name not in matched_inf_dirs_ukbb:
            angle_files = list(inf_case_dir.glob("sulcus_angles_*.json"))
            if angle_files:
                inf_angles = load_sulcus_angles(angle_files[0])
                if inf_angles:
                    extra_inf_ukbb.append({
                        "case_inf": inf_case_dir.name,
                        "inf_angles_full": inf_angles,
                        "num_slices_inf": len(inf_angles),
                    })
    if extra_inf_fm:
        print(f"Extra inference-only fastMRI cases: {len(extra_inf_fm)}")
    if extra_inf_ukbb:
        print(f"Extra inference-only UKBB cases: {len(extra_inf_ukbb)}")

    # ── Load inference curves ────────────────────────────────────────────
    def load_all_curves(
        inf_dir: Path,
        min_angle: Optional[float] = None,
        max_jump: Optional[float] = None,
        restrict_dirs: Optional[set] = None,
    ) -> Tuple[List[np.ndarray], int, int]:
        json_files = sorted(inf_dir.rglob("sulcus_angles_*.json"))
        if restrict_dirs is not None:
            json_files = [jf for jf in json_files
                          if jf.parent in restrict_dirs]
        curves, excluded = [], 0
        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    data = json.load(f)
                angles = data.get("angles", [])
                if not angles:
                    continue
                rs = _resample_angles(angles, num_pts,
                                      min_angle=min_angle,
                                      max_jump=max_jump)
                if rs is not None:
                    curves.append(rs)
                else:
                    excluded += 1
            except Exception:
                pass
        return curves, len(json_files), excluded

    # For Plot 2 curves: use whitelist-restricted indexes if available
    restrict_fm  = {v for v in idx_11b.values()} if wl_fastmri else None
    restrict_ukbb_p2 = matched_ukbb_dirs if wl_ukbb is None else {v for v in idx_11c.values()}

    curves_11b, total_11b, excl_11b = load_all_curves(
        inf_11b, min_angle=args.min_angle, max_jump=args.max_jump,
        restrict_dirs=restrict_fm)
    curves_11c, total_11c, excl_11c = load_all_curves(
        inf_11c, min_angle=args.min_angle,
        restrict_dirs=restrict_ukbb_p2)
    print(f"Curves 11b: {len(curves_11b)}/{total_11b} (excl {excl_11b})")
    print(f"Curves 11c: {len(curves_11c)}/{total_11c}")

    # ══════════════════════════════════════════════════════════════════════
    #  PLOT 1 – Boxplot of deviations
    # ══════════════════════════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    dev_data, box_labels, box_colors = [], [], []
    if fastmri_recs:
        dev_data.append([r["deviation"] for r in fastmri_recs])
        box_labels.append(f"fastMRI (11b)\nn = {len(fastmri_recs)}")
        box_colors.append("tab:blue")
    if ukbb_recs:
        dev_data.append([r["deviation"] for r in ukbb_recs])
        box_labels.append(f"UKBB (11c)\nn = {len(ukbb_recs)}")
        box_colors.append("tab:red")

    bp = ax1.boxplot(dev_data, labels=box_labels, patch_artist=True,
                     widths=0.45, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor="black", markersize=6))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    # Overlay individual points
    for i, (devs, col) in enumerate(zip(dev_data, box_colors)):
        x_jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(devs))
        ax1.scatter(np.full(len(devs), i + 1) + x_jitter, devs,
                    color=col, edgecolors="k", linewidths=0.4,
                    s=40, zorder=5, alpha=0.8)

    ax1.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Deviation from Manual Angle (deg)", fontsize=12)
    ax1.set_title("Inference vs. Manual Sulcus Angle Deviation", fontsize=13)
    ax1.grid(axis="y", alpha=0.3)

    # Add stats text
    stats_parts = []
    for label, devs in zip(box_labels, dev_data):
        d = np.array(devs)
        stats_parts.append(f"{label.split(chr(10))[0]}: "
                           f"mean={np.mean(d):+.1f} deg, "
                           f"median={np.median(d):+.1f} deg, "
                           f"std={np.std(d):.1f} deg")
    fig1.text(0.02, 0.01, "  |  ".join(stats_parts),
              fontsize=8, ha="left", va="bottom",
              bbox=dict(facecolor="white", alpha=0.8, edgecolor="silver"))

    fig1.tight_layout()
    out1 = f"{args.output_prefix}_boxplot.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved boxplot:   {out1}")

    # ══════════════════════════════════════════════════════════════════════
    #  PLOT 2 – Continuous curves + manual scatter
    # ══════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    x_vals = np.linspace(0.0, 100.0, num_pts)

    def plot_curves(curves, color, label, show_individual=False,
                    individual_alpha=0.15):
        if not curves:
            return
        # Individual curves (faint)
        if show_individual:
            for c in curves:
                ax2.plot(x_vals, c, color=color, alpha=individual_alpha,
                         linewidth=0.5)
        stacked = np.vstack(curves)
        mean_v = np.nanmean(stacked, axis=0)
        std_v  = np.nanstd(stacked, axis=0)
        ax2.plot(x_vals, mean_v, color=color, linewidth=2.5,
                 label=f"{label} mean (n={len(curves)})")
        ax2.fill_between(x_vals, mean_v - std_v, mean_v + std_v,
                         color=color, alpha=0.15, label=f"{label} +/-1 std")

    plot_curves(curves_11b, "tab:blue", "fastMRI (11b)")
    plot_curves(curves_11c, "tab:red",  "UKBB (11c)",
                show_individual=True, individual_alpha=0.15)

    # Overlay manual measurements
    if fastmri_recs:
        ax2.scatter([r["norm_position"] for r in fastmri_recs],
                    [r["manual_angle"] for r in fastmri_recs],
                    marker="o", s=70, color="tab:blue", edgecolors="black",
                    linewidths=0.8, zorder=10,
                    label=f"Manual fastMRI (n={len(fastmri_recs)})")
    if ukbb_recs:
        ax2.scatter([r["norm_position"] for r in ukbb_recs],
                    [r["manual_angle"] for r in ukbb_recs],
                    marker="s", s=70, color="tab:red", edgecolors="black",
                    linewidths=0.8, zorder=10,
                    label=f"Manual UKBB (n={len(ukbb_recs)})")

    # Extend x-axis if any manual points are outside [0, 100]
    all_pos = [r["norm_position"] for r in records]
    x_lo = min(0, min(all_pos) - 2)
    x_hi = max(100, max(all_pos) + 2)
    ax2.set_xlim(x_lo, x_hi)

    ax2.set_xlabel("Normalized Slice Position (%)", fontsize=12)
    ax2.set_ylabel("Sulcus Angle (degrees)", fontsize=12)
    ax2.set_title("Sulcus Angle: Inference Curves + Manual Measurements", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)

    # Add stats text box
    stats_lines2 = [
        f"Filters: min_angle={args.min_angle}, max_jump={args.max_jump}",
        f"fastMRI (11b): {len(curves_11b)}/{total_11b} cases",
        f"UKBB (11c): {len(curves_11c)} matched cases",
    ]
    fig2.text(0.02, 0.02, "\n".join(stats_lines2), ha="left", va="bottom",
              fontsize=8, bbox=dict(facecolor="white", alpha=0.8, edgecolor="silver"))

    fig2.tight_layout()
    out2 = f"{args.output_prefix}_curves.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved curves:    {out2}")

    # ══════════════════════════════════════════════════════════════════════
    #  PLOT 3 – Boxplot of actual manual sulcus angles
    # ══════════════════════════════════════════════════════════════════════
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    angle_data, angle_labels, angle_colors = [], [], []
    if fastmri_recs:
        angle_data.append([r["manual_angle"] for r in fastmri_recs])
        angle_labels.append(f"fastMRI\nn = {len(fastmri_recs)}")
        angle_colors.append("tab:blue")
    if ukbb_recs:
        angle_data.append([r["manual_angle"] for r in ukbb_recs])
        angle_labels.append(f"In-house TD\nn = {len(ukbb_recs)}")
        angle_colors.append("tab:red")

    bp3 = ax3.boxplot(angle_data, labels=angle_labels, patch_artist=True,
                      widths=0.45, showmeans=True,
                      meanprops=dict(marker="D", markerfacecolor="white",
                                     markeredgecolor="black", markersize=6))
    for patch, color in zip(bp3["boxes"], angle_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    # Overlay individual points
    for i, (vals, col) in enumerate(zip(angle_data, angle_colors)):
        x_jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax3.scatter(np.full(len(vals), i + 1) + x_jitter, vals,
                    color=col, edgecolors="k", linewidths=0.4,
                    s=40, zorder=5, alpha=0.8)

    ax3.set_ylabel("Manual Sulcus Angle (deg)", fontsize=12)
    ax3.set_title("Manual Sulcus Angle Measurements", fontsize=13)
    ax3.grid(axis="y", alpha=0.3)

    # Add stats text
    stats_parts3 = []
    for label, vals in zip(angle_labels, angle_data):
        v = np.array(vals)
        stats_parts3.append(f"{label.split(chr(10))[0]}: "
                            f"mean={np.mean(v):.1f} deg, "
                            f"median={np.median(v):.1f} deg, "
                            f"std={np.std(v):.1f} deg")
    fig3.text(0.02, 0.01, "  |  ".join(stats_parts3),
              fontsize=8, ha="left", va="bottom",
              bbox=dict(facecolor="white", alpha=0.8, edgecolor="silver"))

    fig3.tight_layout()
    out3 = f"{args.output_prefix}_manual_angles.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved manual angles: {out3}")

    # ══════════════════════════════════════════════════════════════════════
    #  PLOT 4 – Per-case UKBB plots (one per patient)
    # ══════════════════════════════════════════════════════════════════════
    per_case_dir = Path(args.output_prefix + "_ukbb_per_case")
    per_case_dir.mkdir(exist_ok=True)

    for i, rec in enumerate(sorted(ukbb_recs, key=lambda r: r["case_med"])):
        fig4, ax4 = plt.subplots(figsize=(8, 5))

        # Inference curve
        angles_full = rec["inf_angles_full"]
        n_sl = rec["num_slices_inf"]
        x_curve = np.linspace(0.0, 100.0, n_sl)
        ax4.plot(x_curve, angles_full, color="tab:red", linewidth=2,
                 label="Inference curve")

        # Manual measurement point
        ax4.scatter(rec["norm_position"], rec["manual_angle"],
                    marker="s", s=120, color="tab:green", edgecolors="black",
                    linewidths=1.2, zorder=10, label="Manual measurement")

        # Vertical line at the manual slice position
        ax4.axvline(rec["norm_position"], color="tab:green", linewidth=1.2,
                     linestyle="--", alpha=0.7,
                     label=f"Manual slice pos = {rec['norm_position']:.1f}%")

        # Annotate deviation
        ax4.annotate(
            f"dev = {rec['deviation']:+.1f} deg\n"
            f"dz = {rec['z_distance_mm']:.2f} mm",
            xy=(rec["norm_position"], rec["inference_angle"]),
            xytext=(10, -30), textcoords="offset points",
            fontsize=9, color="black",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="silver"),
        )

        # Short case ID for title
        case_short = rec["case_med"][:8]
        ax4.set_xlabel("Normalized Slice Position (%)", fontsize=11)
        ax4.set_ylabel("Sulcus Angle (degrees)", fontsize=11)
        ax4.set_title(f"UKBB {case_short}  |  inf={rec['inference_angle']:.1f} deg  "
                       f"manual={rec['manual_angle']:.1f} deg  "
                       f"(dev={rec['deviation']:+.1f} deg)", fontsize=10)
        ax4.set_ylim(90, 180)
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="best", fontsize=9)
        fig4.tight_layout()

        out4 = per_case_dir / f"{i+1:02d}_{case_short}.png"
        fig4.savefig(out4, dpi=120, bbox_inches="tight")
        plt.close(fig4)

    print(f"Saved {len(ukbb_recs)} per-case UKBB plots in: {per_case_dir}")

    # ══════════════════════════════════════════════════════════════════════
    #  PLOT 4b – 3 example TD patients combined on one plot
    # ══════════════════════════════════════════════════════════════════════
    example_ids = ["0000DF2B", "000012C8", "00007D44"]
    example_labels = ["Example Patient TD 1", "Example Patient TD 2", "Example Patient TD 3"]
    # 3 maximally distinct red-family hues: burgundy, vermillion, rose-pink
    example_colors = ["#7B1F3A", "#E03020", "#F28C8C"]
    example_marker_colors = ["#7B1F3A", "#E03020", "#F28C8C"]  # matching line & marker

    example_recs = []
    for eid in example_ids:
        for rec in ukbb_recs:
            if rec["case_med"].upper().startswith(eid):
                example_recs.append(rec)
                break

    if len(example_recs) == 3:
        # --- Version 1: single combined plot ---
        fig4b, ax4b = plt.subplots(figsize=(10, 6))
        for rec, label, col, mcol in zip(example_recs, example_labels,
                                          example_colors, example_marker_colors):
            angles_full = rec["inf_angles_full"]
            n_sl = rec["num_slices_inf"]
            x_curve = np.linspace(0.0, 100.0, n_sl)
            ax4b.plot(x_curve, angles_full, color=col, linewidth=2.2,
                      label=f"{label} (SA profile)")
            ax4b.scatter(rec["norm_position"], rec["manual_angle"],
                         marker="s", s=100, color=mcol, edgecolors="black",
                         linewidths=1.0, zorder=10,
                         label=f"{label} (manual)")
            ax4b.plot([rec["norm_position"], rec["norm_position"]],
                      [rec["inference_angle"], rec["manual_angle"]],
                      color=col, linewidth=1.2, alpha=0.6, linestyle="--",
                      zorder=9)

        ax4b.set_xlabel("Normalized Slice Position [NSP] (%)", fontsize=15, fontweight='bold')
        ax4b.set_ylabel("Sulcus Angle (\u00b0)", fontsize=15, fontweight='bold')
        ax4b.set_title("Example SA Profiles", fontsize=16, fontweight='bold')
        ax4b.set_ylim(90, 180)
        ax4b.grid(True, alpha=0.3)
        ax4b.legend(loc="best", fontsize=10)
        fig4b.tight_layout()
        out4b = f"{args.output_prefix}_example_td_combined.pdf"
        fig4b.savefig(out4b, bbox_inches="tight")
        plt.close(fig4b)
        print(f"Saved example TD combined: {out4b}")

        # --- Version 2: 3 side-by-side subplots ---
        fig4c, axes4c = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for ax, rec, label, col, mcol in zip(axes4c, example_recs, example_labels,
                                              example_colors, example_marker_colors):
            angles_full = rec["inf_angles_full"]
            n_sl = rec["num_slices_inf"]
            x_curve = np.linspace(0.0, 100.0, n_sl)
            ax.plot(x_curve, angles_full, color=col, linewidth=2.2,
                    label="SA profile")
            ax.scatter(rec["norm_position"], rec["manual_angle"],
                       marker="s", s=100, color=mcol, edgecolors="black",
                       linewidths=1.0, zorder=10, label="Manual")
            ax.plot([rec["norm_position"], rec["norm_position"]],
                    [rec["inference_angle"], rec["manual_angle"]],
                    color=col, linewidth=1.2, alpha=0.6, linestyle="--",
                    zorder=9)
            ax.set_xlabel("Normalized Slice Position [NSP] (%)", fontsize=13, fontweight='bold')
            ax.set_title(label, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=9)

        axes4c[0].set_ylabel("Sulcus Angle (\u00b0)", fontsize=13, fontweight='bold')
        for ax in axes4c:
            ax.set_ylim(90, 180)
        fig4c.suptitle("Example SA Profiles", fontsize=16, fontweight='bold')
        fig4c.tight_layout()
        out4c = f"{args.output_prefix}_example_td_sidebyside.pdf"
        fig4c.savefig(out4c, bbox_inches="tight")
        plt.close(fig4c)
        print(f"Saved example TD side-by-side: {out4c}")
    else:
        print(f"WARNING: Could not find all 3 example TD cases "
              f"(found {len(example_recs)}/{len(example_ids)})")

    # ══════════════════════════════════════════════════════════════════════
    #  PLOT 5 – Matched 30+30 inference curves + manual scatter
    # ══════════════════════════════════════════════════════════════════════
    fig5, ax5 = plt.subplots(figsize=(12, 7))
    x_vals5 = np.linspace(0.0, 100.0, num_pts)

    # Resample matched inference curves from records + extra inference-only cases
    matched_curves_fm = []
    for rec in fastmri_recs + extra_inf_fm:
        a = rec["inf_angles_full"]
        if len(a) > 1:
            rs = np.interp(x_vals5,
                           np.linspace(0, 100, len(a)),
                           np.array(a, dtype=np.float64))
            matched_curves_fm.append(rs)

    matched_curves_ukbb = []
    for rec in ukbb_recs + extra_inf_ukbb:
        a = rec["inf_angles_full"]
        if len(a) > 1:
            rs = np.interp(x_vals5,
                           np.linspace(0, 100, len(a)),
                           np.array(a, dtype=np.float64))
            matched_curves_ukbb.append(rs)

    def plot5_group(curves, color, label, show_individual=True):
        if not curves:
            return
        if show_individual:
            for c in curves:
                ax5.plot(x_vals5, c, color=color, alpha=0.12, linewidth=0.5)
        stacked = np.vstack(curves)
        mean_v = np.nanmean(stacked, axis=0)
        std_v = np.nanstd(stacked, axis=0)
        ax5.plot(x_vals5, mean_v, color=color, linewidth=2.5,
                 label=f"{label} mean (n={len(curves)})")
        ax5.fill_between(x_vals5, mean_v - std_v, mean_v + std_v,
                         color=color, alpha=0.15, label=f"{label} +/-1 std")

    plot5_group(matched_curves_fm,   "tab:blue", "fastMRI", show_individual=True)
    plot5_group(matched_curves_ukbb, "tab:red",  "In-house TD", show_individual=True)

    # Manual scatter points + vertical deviation lines
    for recs, color, marker, mlabel in [
        (fastmri_recs, "tab:blue", "o", "Manual fastMRI"),
        (ukbb_recs,    "tab:red",  "s", "Manual in-house TD"),
    ]:
        if not recs:
            continue
        # Vertical lines from inference to manual at each patient's slice pos
        for r in recs:
            ax5.plot([r["norm_position"], r["norm_position"]],
                     [r["inference_angle"], r["manual_angle"]],
                     color=color, linewidth=1.2, alpha=0.6, zorder=9)
        # Manual measurement scatter
        ax5.scatter([r["norm_position"] for r in recs],
                    [r["manual_angle"] for r in recs],
                    marker=marker, s=70, color=color, edgecolors="black",
                    linewidths=0.8, zorder=10,
                    label=f"{mlabel} (n={len(recs)})")

    # Extend x-axis if needed
    all_pos5 = [r["norm_position"] for r in records]
    ax5.set_xlim(min(0, min(all_pos5) - 2), max(100, max(all_pos5) + 2))

    ax5.set_xlabel("Normalized Slice Position [NSP] (%)", fontsize=15, fontweight='bold')
    ax5.set_ylabel("Sulcus Angle (\u00b0)", fontsize=15, fontweight='bold')
    ax5.set_title("SA Profiles with corresponding manual measurement", fontsize=16, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc="best", fontsize=11)

    fig5.tight_layout()
    out5 = f"{args.output_prefix}_matched_curves.pdf"
    fig5.savefig(out5, bbox_inches="tight")
    plt.close(fig5)
    print(f"Saved matched curves: {out5}")

    # ══════════════════════════════════════════════════════════════════════
    #  PLOT 6 – Inference-only mean+std for the matched 30+30
    # ══════════════════════════════════════════════════════════════════════
    fig6, ax6 = plt.subplots(figsize=(12, 7))

    def plot6_group(curves, color, label):
        if not curves:
            return
        stacked = np.vstack(curves)
        mean_v = np.nanmean(stacked, axis=0)
        std_v  = np.nanstd(stacked, axis=0)
        ax6.plot(x_vals5, mean_v, color=color, linewidth=2.5,
                 label=f"{label} mean (n={len(curves)})")
        ax6.fill_between(x_vals5, mean_v - std_v, mean_v + std_v,
                         color=color, alpha=0.18,
                         label=f"{label} +/-1 std")

    plot6_group(matched_curves_fm,   "tab:blue", "fastMRI")
    plot6_group(matched_curves_ukbb, "tab:red",  "In-house TD")

    ax6.set_xlim(0, 100)
    ax6.set_xlabel("Normalized Slice Position (%)", fontsize=12)
    ax6.set_ylabel("Sulcus Angle (degrees)", fontsize=12)
    ax6.set_title("Matched Cases: Inference Mean +/- Std", fontsize=14)
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc="best", fontsize=10)

    fig6.text(0.02, 0.02,
              f"fastMRI: n={len(matched_curves_fm)},  "
              f"In-house TD: n={len(matched_curves_ukbb)}",
              ha="left", va="bottom", fontsize=8,
              bbox=dict(facecolor="white", alpha=0.8, edgecolor="silver"))

    fig6.tight_layout()
    out6 = f"{args.output_prefix}_inference_only.png"
    fig6.savefig(out6, dpi=150, bbox_inches="tight")
    plt.close(fig6)
    print(f"Saved inference-only curves: {out6}")

    # ══════════════════════════════════════════════════════════════════════
    #  PLOT 7 – Combined: manual boxplot (left) + inference curves (right)
    # ══════════════════════════════════════════════════════════════════════
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 11,
    })

    fig7, (ax7l, ax7r) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # --- Left: manual angles boxplot (no individual points) ---
    box_data7, box_labels7, box_colors7 = [], [], []
    if fastmri_recs:
        box_data7.append([r["manual_angle"] for r in fastmri_recs])
        box_labels7.append(f"fastMRI\nn = {len(fastmri_recs)}")
        box_colors7.append("tab:blue")
    if ukbb_recs:
        box_data7.append([r["manual_angle"] for r in ukbb_recs])
        box_labels7.append(f"In-house TD\nn = {len(ukbb_recs)}")
        box_colors7.append("tab:red")

    bp7 = ax7l.boxplot(box_data7, labels=box_labels7, patch_artist=True,
                       widths=0.45, showmeans=True,
                       meanprops=dict(marker="D", markerfacecolor="white",
                                      markeredgecolor="black", markersize=6))
    for patch, color in zip(bp7["boxes"], box_colors7):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    ax7l.set_ylabel("Sulcus Angle (\u00b0)", fontsize=15, fontweight='bold')
    ax7l.set_title("Manual Measurements", fontsize=16, fontweight='bold')
    for lbl in ax7l.get_xticklabels():
        lbl.set_fontweight('bold')
    ax7l.grid(axis="y", alpha=0.3)

    # --- Right: inference mean +/- std curves ---
    def plot7_group(curves, color, label):
        if not curves:
            return
        stacked = np.vstack(curves)
        mean_v = np.nanmean(stacked, axis=0)
        std_v  = np.nanstd(stacked, axis=0)
        ax7r.plot(x_vals5, mean_v, color=color, linewidth=2.5,
                  label=f"{label} mean (n={len(curves)})")
        ax7r.fill_between(x_vals5, mean_v - std_v, mean_v + std_v,
                          color=color, alpha=0.18,
                          label=f"{label} \u00b11 std")

    plot7_group(matched_curves_fm,   "tab:blue", "fastMRI")
    plot7_group(matched_curves_ukbb, "tab:red",  "In-house TD")

    ax7r.set_xlim(0, 100)
    ax7r.set_xlabel("Normalized Slice Position [NSP] (%)", fontsize=15, fontweight='bold')
    ax7r.set_title("Automated SA Profiles", fontsize=16, fontweight='bold')
    ax7r.grid(True, alpha=0.3)
    ax7r.legend(loc="best", fontsize=11)

    fig7.tight_layout()
    out7 = f"{args.output_prefix}_combined.pdf"
    fig7.savefig(out7, bbox_inches="tight")
    plt.close(fig7)
    print(f"Saved combined plot: {out7}")

    # Print per-case table
    print("\n" + "=" * 100)
    print(f"{'Dataset':<18} {'Case (med eval)':<55} {'Manual':>7} {'Infer':>7} {'Dev':>7} {'Pos%':>6}")
    print("-" * 100)
    for r in sorted(records, key=lambda x: (x["dataset"], x["case_med"])):
        print(f"{r['dataset']:<18} {r['case_med'][:54]:<55} "
              f"{r['manual_angle']:7.1f} {r['inference_angle']:7.1f} "
              f"{r['deviation']:+7.1f} {r['norm_position']:6.1f}")
    print("=" * 100)

    # Print summary statistics table
    hdr = (f"{'':14s} {'Manual Measurement':>24s}  {'Automated SA Profile':>24s}"
           f"  {'MAE':>8} {'MAE Std':>8} {'Mean Slice':>11} {'n':>5}")
    sub = (f"{'':14s} {'Mean':>12} {'Std Dev':>11}  {'Mean':>12} {'Std Dev':>11}"
           f"  {'':>8} {'':>8} {'Pos.%':>11} {'':>5}")
    width = len(hdr) + 2
    print("\n" + "=" * width)
    print(hdr)
    print(sub)
    print("-" * width)
    for label, recs in [("fastMRI", fastmri_recs),
                         ("in-house TD", ukbb_recs),
                         ("All", records)]:
        if not recs:
            continue
        manuals = np.array([r["manual_angle"] for r in recs])
        infers  = np.array([r["inference_angle"] for r in recs])
        devs    = np.array([r["deviation"] for r in recs])
        positions = np.array([r["norm_position"] for r in recs])
        n = len(recs)
        abs_devs = np.abs(devs)
        mae = np.mean(abs_devs)
        mae_std = np.std(abs_devs)
        deg = "\u00b0"
        print(f"{label:<14s} {np.mean(manuals):12.1f}{deg} {np.std(manuals):10.1f}{deg}"
              f"  {np.mean(infers):12.1f}{deg} {np.std(infers):10.1f}{deg}"
              f"  {mae:7.1f}{deg} {mae_std:7.1f}{deg} {np.mean(positions):10.1f}% {n:5d}")
    print("=" * width)


if __name__ == "__main__":
    main()
