"""
Generate a detailed per-case figure for a UKBB patient.

Top row:  Raw axial slice (low-res) with manual trochlea points and angle,
          plus the SA profile curve.
Bottom:   Several super-resolution slices with heatmap overlay, keypoints,
          and inferred sulcus angle.

Usage:
  python scripts/detailed_case_figure.py --case_id DF2B --output detailed_DF2B.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib

import matplotlib
try:
    matplotlib.use("Agg")
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.gridspec import GridSpec
from scipy.ndimage import zoom as scipy_zoom


# ─── helpers ─────────────────────────────────────────────────────────────────

def compute_angle_deg_2d(p1, p2, p3):
    """Angle at p2 between rays p2->p1 and p2->p3 (degrees)."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def draw_angle_arc(ax, p1, p2, p3, radius=20, color="yellow", lw=2,
                   draw_rays=True, fontsize=0):
    """Draw an arc at p2 indicating the angle, plus the two rays."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle1 = np.degrees(np.arctan2(v1[1], v1[0]))
    angle2 = np.degrees(np.arctan2(v2[1], v2[0]))

    if draw_rays:
        ax.plot([p2[0], p1[0]], [p2[1], p1[1]], color=color, linewidth=lw, zorder=8)
        ax.plot([p2[0], p3[0]], [p2[1], p3[1]], color=color, linewidth=lw, zorder=8)

    # Draw arc (smaller angle)
    a1, a2 = sorted([angle1, angle2])
    if a2 - a1 > 180:
        a1, a2 = a2, a1 + 360
    arc = Arc(p2, 2 * radius, 2 * radius, angle=0,
              theta1=a1, theta2=a2, color=color, linewidth=lw, zorder=8)
    ax.add_patch(arc)

    if fontsize > 0:
        angle_val = compute_angle_deg_2d(p1, p2, p3)
        mid_angle = np.radians((a1 + a2) / 2)
        tx = p2[0] + radius * 1.5 * np.cos(mid_angle)
        ty = p2[1] + radius * 1.5 * np.sin(mid_angle)
        ax.text(tx, ty, f"{angle_val:.1f}\u00b0", fontsize=fontsize,
                color=color, fontweight="bold", ha="center", va="center", zorder=12,
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1))


def main():
    parser = argparse.ArgumentParser(description="Detailed per-case figure")
    parser.add_argument("--case_id", type=str, default="DF2B",
                        help="Short UKBB case ID (e.g. DF2B)")
    parser.add_argument("--raw_nifti_dir", type=str,
                        default="Medical_Evaluation/Noel_Medical_Evaluation_Classic_Style")
    parser.add_argument("--med_eval_dir", type=str,
                        default=r"Medical_Evaluation\Knee_medical_evaluation_out\Knee out")
    parser.add_argument("--sr_nifti_dir", type=str,
                        default="11_c_INR_muon_results_UKBB")
    parser.add_argument("--inference_dir", type=str,
                        default="inference_output_11c_w_plots")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--bottom_slices", type=int, default=6,
                        help="Number of slices to show in bottom row")
    args = parser.parse_args()

    cid = args.case_id.upper()
    if args.output is None:
        args.output = f"detailed_{cid}.png"

    # ── Locate files ──────────────────────────────────────────────────────
    raw_nifti = next((f for f in Path(args.raw_nifti_dir).glob("*.nii.gz")
                      if cid in f.name.upper()), None)
    if raw_nifti is None:
        raise FileNotFoundError(f"Raw NIfTI not found for {cid}")

    med_case = next((d for d in Path(args.med_eval_dir).iterdir()
                     if d.is_dir() and cid in d.name.upper()), None)
    if med_case is None:
        raise FileNotFoundError(f"Medical eval case not found for {cid}")

    sr_nifti = next((f for f in Path(args.sr_nifti_dir).glob("*.nii.gz")
                     if cid in f.name.upper()), None)
    if sr_nifti is None:
        raise FileNotFoundError(f"SR NIfTI not found for {cid}")

    inf_case = next((d for d in Path(args.inference_dir).iterdir()
                     if d.is_dir() and cid in d.name.upper()), None)
    if inf_case is None:
        raise FileNotFoundError(f"Inference dir not found for {cid}")

    print(f"Raw: {raw_nifti.name}")
    print(f"SR:  {sr_nifti.name}")
    print(f"Inf: {inf_case.name}")

    # ── Load data ─────────────────────────────────────────────────────────
    raw_img = nib.load(str(raw_nifti))
    raw_vol = raw_img.get_fdata()
    raw_affine = raw_img.affine

    sr_img = nib.load(str(sr_nifti))
    sr_vol = sr_img.get_fdata()
    sr_affine = sr_img.affine

    print(f"Raw: {raw_vol.shape}, zooms={[round(z,3) for z in raw_img.header.get_zooms()]}")
    print(f"SR:  {sr_vol.shape}, zooms={[round(z,3) for z in sr_img.header.get_zooms()]}")

    # Manual trochlea points (LPS from Slicer markup -> convert to RAS for NIfTI)
    with open(med_case / "TrochleaPoints.mrk.json") as f:
        trk_data = json.load(f)
    cps = trk_data["markups"][0]["controlPoints"][-3:]
    pts_lps = [np.array(cp["position"], dtype=np.float64) for cp in cps]
    pts_ras = [np.array([-p[0], -p[1], p[2]]) for p in pts_lps]

    # Map to raw voxel coordinates
    inv_raw = np.linalg.inv(raw_affine)
    pts_raw_vox = [(inv_raw @ np.append(p, 1.0))[:3] for p in pts_ras]
    manual_z_raw = int(round(np.mean([v[2] for v in pts_raw_vox])))
    manual_z_raw = np.clip(manual_z_raw, 0, raw_vol.shape[2] - 1)

    # Pixel coords on raw axial slice: [col=i, row=j] for imshow(T)
    pts_raw_xy = [(v[0], v[1]) for v in pts_raw_vox]

    manual_angle = compute_angle_deg_2d(pts_lps[0][:2], pts_lps[1][:2], pts_lps[2][:2])
    print(f"Manual SA: {manual_angle:.1f} deg  (raw slice z={manual_z_raw})")

    # Load heatmaps
    hm_data = np.load(str(inf_case / "axial_heatmaps.npz"))
    heatmaps = hm_data["heatmaps"]
    z_indices = hm_data["z_indices"]

    # Load sulcus angles + fitted points
    angle_json = list(inf_case.glob("sulcus_angles_*.json"))[0]
    with open(angle_json) as f:
        sa_data = json.load(f)
    inf_angles = sa_data["angles"]
    fitted_pts = sa_data["fitted_points"]
    N_slices = len(inf_angles)

    # Map manual points to SR voxel space -> find heatmap slice
    inv_sr = np.linalg.inv(sr_affine)
    manual_sr_vox = [(inv_sr @ np.append(p, 1.0))[:3] for p in pts_ras]
    manual_z_sr = np.mean([v[2] for v in manual_sr_vox])
    manual_hm_idx = int(np.argmin(np.abs(z_indices - round(manual_z_sr))))
    inf_at_manual = inf_angles[manual_hm_idx]
    print(f"Manual z_SR={manual_z_sr:.1f} -> hm idx={manual_hm_idx}, "
          f"inference={inf_at_manual:.1f} deg, dev={inf_at_manual - manual_angle:+.1f} deg")

    # ── Select slices for bottom row ──────────────────────────────────────
    n_bottom = args.bottom_slices
    bottom_indices = np.linspace(0, N_slices - 1, n_bottom, dtype=int).tolist()
    # Include manual slice
    closest_to_manual = min(range(len(bottom_indices)),
                           key=lambda i: abs(bottom_indices[i] - manual_hm_idx))
    bottom_indices[closest_to_manual] = manual_hm_idx
    bottom_indices = sorted(set(bottom_indices))
    # Fill gaps if needed
    while len(bottom_indices) < n_bottom:
        remaining = set(range(N_slices)) - set(bottom_indices)
        if not remaining:
            break
        best = max(remaining, key=lambda x: min(abs(x - b) for b in bottom_indices))
        bottom_indices.append(best)
        bottom_indices = sorted(bottom_indices)
    print(f"Bottom slices: {bottom_indices}")

    # ══════════════════════════════════════════════════════════════════════
    #  CREATE FIGURE
    # ══════════════════════════════════════════════════════════════════════
    n_bottom = len(bottom_indices)
    fig = plt.figure(figsize=(3.5 * n_bottom, 9))
    gs = GridSpec(2, n_bottom, figure=fig, height_ratios=[1.1, 1],
                  hspace=0.25, wspace=0.08)

    # ── TOP-LEFT: Raw axial slice with manual angle ──────────────────────
    ax_raw = fig.add_subplot(gs[0, : n_bottom // 2])

    raw_slice = raw_vol[:, :, manual_z_raw].T  # transpose -> (j, i) for display
    ax_raw.imshow(raw_slice, cmap="gray", origin="lower", aspect="equal")

    # Crop around the trochlea region
    all_x = [p[0] for p in pts_raw_xy]
    all_y = [p[1] for p in pts_raw_xy]
    cx, cy = np.mean(all_x), np.mean(all_y)
    span = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
    margin = span * 1.2
    ax_raw.set_xlim(cx - margin, cx + margin)
    ax_raw.set_ylim(cy - margin, cy + margin)

    # Label points T1, T2, T3
    labels_pts = ["T1", "T2", "T3"]
    for idx, (px, py) in enumerate(pts_raw_xy):
        offsets = [(-12, 10), (5, -15), (10, 10)]
        ax_raw.annotate(labels_pts[idx], (px, py), fontsize=10, color="#2196F3",
                       xytext=offsets[idx], textcoords="offset points",
                       fontweight="bold", zorder=11,
                       bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1))

    # Draw angle arc (red lines and arc)
    draw_angle_arc(ax_raw, pts_raw_xy[0], pts_raw_xy[1], pts_raw_xy[2],
                   radius=max(span * 0.25, 15), color="red", lw=2.5, fontsize=11)

    ax_raw.set_title(f"Raw Axial Slice\n"
                     f"Manual SA = {manual_angle:.1f}\u00b0",
                     fontsize=13, fontweight="bold", pad=8)
    ax_raw.axis("off")

    # ── TOP-RIGHT: SA profile curve ──────────────────────────────────────
    ax_curve = fig.add_subplot(gs[0, n_bottom // 2:])

    x_norm = np.linspace(0, 100, N_slices)
    ax_curve.plot(x_norm, inf_angles, color="tab:red", linewidth=2.5,
                  label="Automated SA profile", zorder=5)
    ax_curve.fill_between(x_norm, inf_angles, alpha=0.08, color="tab:red")

    # Manual measurement
    manual_norm_pos = manual_hm_idx / (N_slices - 1) * 100
    ax_curve.scatter([manual_norm_pos], [manual_angle], marker="s", s=130,
                    color="#00FF00", edgecolors="black", linewidths=1.5, zorder=10,
                    label=f"Manual = {manual_angle:.1f}\u00b0")
    ax_curve.scatter([manual_norm_pos], [inf_at_manual], marker="^", s=110,
                    color="tab:red", edgecolors="black", linewidths=1.5, zorder=10,
                    label=f"Automated = {inf_at_manual:.1f}\u00b0")
    # Deviation line
    ax_curve.plot([manual_norm_pos, manual_norm_pos],
                 [manual_angle, inf_at_manual],
                 color="gray", linewidth=2, linestyle="--", zorder=9)
    dev = inf_at_manual - manual_angle
    mid_y = (manual_angle + inf_at_manual) / 2
    ax_curve.annotate(f"\u0394 = {dev:+.1f}\u00b0",
                     (manual_norm_pos + 2, mid_y), fontsize=10, color="gray",
                     fontweight="bold")

    # Mark bottom-row slices
    for bi in bottom_indices:
        bi_pos = bi / (N_slices - 1) * 100
        ax_curve.axvline(bi_pos, color="cornflowerblue", alpha=0.3, linewidth=1, zorder=1)

    ax_curve.set_xlabel("Normalized Slice Position [NSP] (%)", fontsize=11)
    ax_curve.set_ylabel("Sulcus Angle [SA] (\u00b0)", fontsize=11)
    ax_curve.set_title("Automated SA Profile", fontsize=13, fontweight="bold", pad=8)
    ax_curve.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_curve.grid(True, alpha=0.3)
    ax_curve.set_ylim(90, 180)
    ax_curve.set_xlim(-2, 102)

    # ── BOTTOM ROW: SR slices with heatmap + keypoints ───────────────────
    for col_idx, sl_idx in enumerate(bottom_indices):
        ax = fig.add_subplot(gs[1, col_idx])

        z_vol = z_indices[sl_idx]

        # Reconstruct the 256x256 image that was fed to the model
        sr_slice_raw = sr_vol[:, :, z_vol]
        sr_slice_rot = np.rot90(sr_slice_raw, k=1)
        scale = np.array([256, 256]) / np.array(sr_slice_rot.shape)
        sr_slice_256 = scipy_zoom(sr_slice_rot, scale, order=1)
        vmin, vmax = sr_slice_256.min(), sr_slice_256.max()
        if vmax > vmin:
            sr_slice_256 = (sr_slice_256 - vmin) / (vmax - vmin)

        # Sum heatmaps for overlay
        hm_sum = heatmaps[sl_idx].sum(axis=0)

        ax.imshow(sr_slice_256, cmap="gray", aspect="equal")
        ax.imshow(hm_sum, cmap="hot", alpha=0.45, aspect="equal",
                  vmin=0, vmax=hm_sum.max() * 0.8)

        # Draw angle between keypoints (red, matching top row)
        p1 = fitted_pts["0"][sl_idx]
        p2 = fitted_pts["1"][sl_idx]
        p3 = fitted_pts["2"][sl_idx]
        draw_angle_arc(ax, p1, p2, p3, radius=14, color="red", lw=2)

        angle_val = inf_angles[sl_idx]
        norm_pos = sl_idx / (N_slices - 1) * 100

        is_manual = (sl_idx == manual_hm_idx)
        if is_manual:
            for spine in ax.spines.values():
                spine.set_edgecolor("lime")
                spine.set_linewidth(4)
            title_color = "#00FF00"
            title_str = f"SA = {angle_val:.1f}\u00b0 | NSP = {norm_pos:.0f}%\n\u25B6 Manual slice"
        else:
            title_color = "black"
            title_str = f"SA = {angle_val:.1f}\u00b0 | NSP = {norm_pos:.0f}%"

        ax.set_title(title_str, fontsize=10,
                    fontweight="bold" if is_manual else "normal",
                    color=title_color)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
