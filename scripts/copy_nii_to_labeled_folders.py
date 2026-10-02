"""
Copy .nii.gz files into matching subfolders in the target directory.

Each .nii.gz file is placed inside the subfolder whose name matches the file's
stem (without the .nii.gz extension). If the subfolder doesn't exist or the
file already exists, it is skipped and reported.

Usage:
    python copy_nii_to_labeled_folders.py
"""

import shutil
from pathlib import Path

SRC_DIR = Path(
    r"W:\07_MasterThesis\04_Medical_LandmarkDetection_Noel"
    r"\Noel_fastMRI_and_UKBB_toLabel_INR_muon_30_01_26"
)
DST_DIR = Path(
    r"W:\07_MasterThesis\04_Medical_LandmarkDetection_Noel"
    r"\Noel_fastMRI_and_UKBB_labeled_23_02_2026_nii_included"
)


def main():
    nii_files = sorted(SRC_DIR.glob("*.nii.gz"))

    if not nii_files:
        print(f"No .nii.gz files found in:\n  {SRC_DIR}")
        return

    copied = 0
    skipped_exists = 0
    skipped_no_folder = []

    for src_file in nii_files:
        # Derive folder name by stripping .nii.gz from the filename
        folder_name = src_file.name.removesuffix(".nii.gz")
        target_folder = DST_DIR / folder_name
        target_path = target_folder / src_file.name

        if not target_folder.exists():
            skipped_no_folder.append(src_file.name)
            continue

        if target_path.exists():
            skipped_exists += 1
            continue

        shutil.copy2(src_file, target_path)
        copied += 1

    # ── Report ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("COPY REPORT")
    print("=" * 60)
    print(f"Total .nii.gz files found : {len(nii_files)}")
    print(f"Copied successfully       : {copied}")
    print(f"Skipped (already exists)  : {skipped_exists}")
    print(f"Skipped (no target folder): {len(skipped_no_folder)}")
    print("=" * 60)

    if skipped_no_folder:
        print(f"\nFiles with MISSING target folders ({len(skipped_no_folder)}):")
        for name in skipped_no_folder:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
