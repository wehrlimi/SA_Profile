import os
import shutil
from pathlib import Path
import re

def main():
    source_root = Path(r"C:\Users\michael\Documents\Data\Sorted_INR_logictesting_V5\fastMRI_nifti\raw")
    target_root = Path(r"W:\07_MasterThesis\04_Medical_LandmarkDetection_Noel\Noel_Medical_Evaluation_Classic_Style")

    # Provided list of fastMRI identifiers from the image
    identifiers = [
        "FB_105209______FB,2950978745_study_39e3351f_res_256_epoch_50_contr_0",
        "FB_106167______FB,8454366124_study_18ace067_res_256_epoch_50_contr_0",
        "FB_107116______FB,1963129082_study_1adeead8_res_256_epoch_50_contr_0",
        "FB_108289______FB,1066653295_study_9e616e57_res_256_epoch_50_contr_0",
        "FB_108448______FB,2864364286_study_07bd1f08_res_256_epoch_50_contr_0",
        "FB_108828______FB,1852509522_study_4a310e9d_res_256_epoch_50_contr_0",
        "FB_111457______FB,3267316790_study_ca5ffd65_res_256_epoch_50_contr_0",
        "FB_111956______FB,1631693799_study_0bddee9e_res_256_epoch_50_contr_0",
        "FB_112874______FB,2669054529_study_e5d27906_res_256_epoch_50_contr_0",
        "FB_112971______FB,9812720504_study_8575823f_res_256_epoch_50_contr_0",
        "FB_114108______FB,2417835292_study_a3ab5a8c_res_256_epoch_50_contr_0",
        "FB_115687______FB,9956335581_study_c4ce1e0f_res_256_epoch_50_contr_0",
        "FB_116262______FB,2343184841_study_1b9e2654_res_256_epoch_50_contr_0",
        "FB_117447______FB,3211091352_study_c2b3b07e_res_256_epoch_50_contr_0",
        "FB_117478______FB,2328004669_study_84c4af94_res_256_epoch_50_contr_0",
        "FB_118447______FB,2327263790_study_c8319dff_res_256_epoch_50_contr_0",
        "FB_119668______FB,2894776846_study_eefbf3c6_res_256_epoch_50_contr_0",
        "FB_119678______FB,3016430853_study_c2432f0b_res_256_epoch_50_contr_0",
        "FB_122796______FB,1199930784_study_14bf4ad2_res_256_epoch_50_contr_0",
        "FB_124868______FB,3173077638_study_8b46caf3_res_256_epoch_50_contr_0",
        "FB_125409______FB,2229272546_study_c2f16ccb_res_256_epoch_50_contr_0",
        "FB_127623______FB,2152859967_study_15f639e2_res_256_epoch_50_contr_0",
        "FB_128397______FB,4401552830_study_bde780f3_res_256_epoch_50_contr_0",
        "FB_129066______FB,2123275683_study_cece71e7_res_256_epoch_50_contr_0",
        "FB_129837______FB,3149712437_study_8a85643b_res_256_epoch_50_contr_0",
        "FB_130297______FB,388913606_study_03a4a4a4_res_256_epoch_50_contr_0",
        "FB_130393______FB,1673680742_study_afc53d3a_res_256_epoch_50_contr_0",
        "FB_130620______FB,1065155164_study_0643c774_res_256_epoch_50_contr_0",
        "FB_131133______FB,1417224736_study_c1eb6a53_res_256_epoch_50_contr_0",
        "FB_132511______FB,1920192301_study_1f728667_res_256_epoch_50_contr_0",
    ]

    if not target_root.exists():
        print(f"Creating target directory: {target_root}")
        target_root.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    missing_dirs = []
    missing_files = []
    skipped_count = 0

    # Get all potential subject folders once to handle underscore variations
    all_subject_folders = list(source_root.glob("FB_*"))

    for idx, identifier in enumerate(identifiers, 1):
        # identifier looks like: FB_105209______FB,2950978745_study_39e3351f_res_256_epoch_50_contr_0
        # split by "_study_"
        match = re.search(r"(FB_.*?)_study_([a-f0-9]+)", identifier)
        if not match:
            print(f"[{idx}/30] ERROR: Could not parse identifier: {identifier}")
            missing_dirs.append(identifier)
            continue
        
        subject_part = match.group(1)
        study_id = match.group(2)
        
        # Normalize subject_part (replace multiple underscores with a single underscore for matching)
        subject_norm = re.sub(r'_+', '_', subject_part)
        
        # Find the actual folder on disk
        target_subject_dir = None
        for folder in all_subject_folders:
            if re.sub(r'_+', '_', folder.name) == subject_norm:
                target_subject_dir = folder
                break
        
        if not target_subject_dir:
            print(f"[{idx}/30] ERROR: Subject folder not found for: {subject_part}")
            missing_dirs.append(identifier)
            continue
            
        study_dir = target_subject_dir / f"study_{study_id}"
        
        if not study_dir.exists():
            print(f"[{idx}/30] ERROR: Study directory not found: {study_dir}")
            missing_dirs.append(identifier)
            continue

        # Find files with TRA or AXIAL (case-insensitive)
        files = list(study_dir.glob("*.nii.gz"))
        matches = [f for f in files if "tra" in f.name.lower() or "axial" in f.name.lower()]

        if not matches:
            print(f"[{idx}/30] WARNING: No TRA or AXIAL file found in {study_dir}")
            missing_files.append(identifier)
            continue

        for source_file in matches:
            dest_file = target_root / source_file.name
            
            if dest_file.exists():
                print(f"[{idx}/30] Skipping (already exists): {source_file.name}")
                skipped_count += 1
                continue

            try:
                shutil.copy2(source_file, dest_file)
                print(f"[{idx}/30] Copied: {source_file.name}")
                copied_count += 1
            except Exception as e:
                print(f"[{idx}/30] FAILED to copy {source_file.name}: {e}")

    print("\n" + "="*30)
    print(f"Results Summary:")
    print(f"Total processed: {len(identifiers)}")
    print(f"Successfully copied: {copied_count}")
    print(f"Skipped (already exists): {skipped_count}")
    
    if missing_dirs:
        print(f"\nMissing Directories ({len(missing_dirs)}):")
        for m in missing_dirs:
            print(f" - {m}")
            
    if missing_files:
        print(f"\nMissing TRA/AXIAL Files in found directories ({len(missing_files)}):")
        for m in missing_files:
            print(f" - {m}")
            
    print("="*30)

if __name__ == "__main__":
    main()
