import boto3
import os
import re
import json
import random
import numpy as np
from omegaconf import OmegaConf
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def is_image_file(filepath):
    """Check if file is a NIfTI image"""
    filename = filepath.split("/")[-1]
    return filename.endswith(".nii.gz") or filename.endswith(".nii")


def is_metadata_file(filepath):
    """Check if file is a Slicer case metadata file"""
    filename = filepath.split("/")[-1]
    return filename == "case_metadata.json"


def load_case_metadata(s3, repo, metadata_path):
    """Load and parse Slicer case metadata"""
    try:
        response = s3.get_object(Bucket=repo, Key=metadata_path)
        content = response['Body'].read().decode('utf-8')
        metadata = json.loads(content)
        return metadata
    except Exception as e:
        print(f"Warning: Could not load metadata from {metadata_path}: {e}")
        return None


def load_axial_labels(s3, repo, labels_path):
    """
    Load axial label points from Slicer markup file.

    Returns:
        list of dicts with 'xy' [x, y] and 'xyz' [x, y, z] coordinates in order (9 points total: 3 per slice)
        Coordinates are in LPS (Left-Posterior-Superior) from Slicer
    """
    try:
        response = s3.get_object(Bucket=repo, Key=labels_path)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)

        control_points = data["markups"][0]["controlPoints"]
        keypoints = []

        for cp in control_points:
            # Extract position (LPS coordinates from Slicer)
            point = np.array(cp["position"])  # [L, P, S] = [x, y, z] in LPS

            # Apply orientation if needed (convert to image space)
            if "orientation" in cp:
                orientation = np.array(cp["orientation"]).reshape(3, 3)
                point = orientation @ point

            # Store both 2D (x, y) for compatibility and full 3D (x, y, z) for slice extraction
            # point[0] = L (left/x), point[1] = P (posterior/y), point[2] = S (superior/z)
            keypoints.append({
                'xy': [point[0], point[1]],  # (x, y) in image space (L, P)
                'xyz': [point[0], point[1], point[2]]  # Full 3D (L, P, S)
            })

        return keypoints

    except Exception as e:
        print(f"Warning: Could not load labels from {labels_path}: {e}")
        return None


def load_boundary_points(s3, repo, bounds_path):
    """
    Load sagittal boundary points (2 points marking trochlea extent).

    Returns:
        dict with 'points' (list of 2 [y, z] coords for compatibility), 'xyz' (list of 2 [x, y, z] full 3D coords),
        and 'z_coords' (list of 2 z values)
        Coordinates are in LPS (Left-Posterior-Superior) from Slicer
    """
    try:
        response = s3.get_object(Bucket=repo, Key=bounds_path)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)

        control_points = data["markups"][0]["controlPoints"]

        if len(control_points) != 2:
            print(f"Warning: Expected 2 boundary points, got {len(control_points)}")
            return None

        points = []  # [y, z] for backward compatibility
        xyz_coords = []  # Full 3D coordinates [x, y, z]
        z_coords = []

        for cp in control_points:
            # Extract position (LPS coordinates from Slicer)
            point = np.array(cp["position"])  # [L, P, S] = [x, y, z] in LPS

            # Apply orientation if needed
            if "orientation" in cp:
                orientation = np.array(cp["orientation"]).reshape(3, 3)
                point = orientation @ point

            # For sagittal view: store (y, z) coordinates for backward compatibility
            # point[0] = L (left/x), point[1] = P (posterior/y), point[2] = S (superior/z)
            points.append([point[1], point[2]])  # (y, z) for sagittal view
            xyz_coords.append([point[0], point[1], point[2]])  # Full 3D (x, y, z)
            z_coords.append(point[2])  # Store z-coordinate separately

        return {
            'points': points,  # [[y, z], [y, z]] for backward compatibility
            'xyz': xyz_coords,  # [[x, y, z], [x, y, z]] full 3D coordinates
            'z_coords': sorted(z_coords)  # [z_min, z_max]
        }

    except Exception as e:
        print(f"Warning: Could not load boundary points from {bounds_path}: {e}")
        return None


def extract_case_data(s3, repo, case_dir_path):
    """
    Extract all data for a single case from Slicer output.

    Returns:
        dict: {
            "image": path_to_image,
            "labels": {
                z_coord_str: {
                    "keypoints": [[x0, y0], [x1, y1], [x2, y2]],  # 3 points per slice [x, y]
                    "xyz_coords": [[x0, y0, z0], ...]  # Full 3D coordinates [x, y, z]
                },
                ...
            },
            "bounds": {
                "points": [[y, z], [y, z]],  # 2 sagittal boundary points (for backward compat)
                "xyz": [[x, y, z], [x, y, z]],  # Full 3D coordinates
                "z_coords": [z_min, z_max]   # sorted z-coordinates
            }
        }
    """
    # List all files in the case directory
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=repo, Prefix=case_dir_path)

    image_path = None
    metadata_path = None
    labels_path = None
    bounds_path = None

    for page in pages:
        if 'Contents' not in page:
            continue

        for obj in page['Contents']:
            key = obj['Key']
            filename = key.split("/")[-1]

            if is_image_file(key):
                image_path = key
            elif filename == "case_metadata.json":
                metadata_path = key
            elif filename == "AxialLabels.mrk.json":
                labels_path = key
            elif filename == "TrochleaBounds.mrk.json":
                bounds_path = key

    # Validate we have all required files
    if not image_path or not metadata_path or not labels_path or not bounds_path:
        print(f"Warning: Incomplete case at {case_dir_path}")
        print(f"  Image: {image_path is not None}")
        print(f"  Metadata: {metadata_path is not None}")
        print(f"  Labels: {labels_path is not None}")
        print(f"  Bounds: {bounds_path is not None}")
        return None

    # Load metadata to get picked slice information
    metadata = load_case_metadata(s3, repo, metadata_path)
    if not metadata:
        return None

    picked_slices = metadata.get("picked_k_slices", [])
    if len(picked_slices) != 3:
        print(f"Warning: Case {case_dir_path} has {len(picked_slices)} slices (expected 3)")
        return None

    # Load boundary points
    bounds = load_boundary_points(s3, repo, bounds_path)
    if not bounds:
        return None

    # Load all axial labels (should be 9 points: 3 per slice)
    all_keypoints = load_axial_labels(s3, repo, labels_path)
    if not all_keypoints:
        return None

    if len(all_keypoints) != 9:
        print(f"Warning: Case {case_dir_path} has {len(all_keypoints)} points (expected 9)")
        return None

    # Organize keypoints by slice using z-coordinates from the actual points
    # Points 0-2 belong to slice 0, 3-5 to slice 1, 6-8 to slice 2
    labels_by_z = {}
    points_per_slice = 3

    for slice_idx in range(3):  # 3 slices
        start_idx = slice_idx * points_per_slice
        end_idx = start_idx + points_per_slice
        slice_keypoints_data = all_keypoints[start_idx:end_idx]

        # Extract [x, y] coordinates for compatibility
        slice_keypoints = [kp['xy'] for kp in slice_keypoints_data]
        
        # Use z-coordinate from the first point in this slice to identify the slice
        # The z-coordinate (S in LPS) corresponds to the axial slice position
        z_coord_from_point = slice_keypoints_data[0]['xyz'][2]
        
        # Convert z-coordinate to slice index (we'll do this later in the dataset using affine)
        # For now, store with the z-coordinate as key (as string for JSON compatibility)
        # But also keep track of full 3D coordinates
        labels_by_z[str(z_coord_from_point)] = {
            'keypoints': slice_keypoints,  # [[x, y], [x, y], [x, y]]
            'xyz_coords': [kp['xyz'] for kp in slice_keypoints_data]  # [[x, y, z], ...] full 3D
        }

    return {
        "image": image_path,
        "labels": labels_by_z,
        "bounds": bounds,
        "metadata": metadata_path
    }


def get_data_paths(cfg):
    """
    Scan S3/LakeFS for Slicer-formatted cases and extract paths.

    Returns:
        dict: {
            case_id: {
                "image": path,
                "labels": {
                    z_coord: [[x, y], [x, y], [x, y]],  # 3 points per slice
                    ...
                },
                "bounds": {
                    "points": [[y, z], [y, z]],  # 2 sagittal points
                    "z_coords": [z_min, z_max]
                }
            }
        }
    """
    s3 = boto3.client('s3', endpoint_url=cfg.lakefs.s3_endpoint, verify=cfg.lakefs.ca_path)
    paginator = s3.get_paginator('list_objects_v2')

    if isinstance(cfg.lakefs.paths, str):
        paths = [cfg.lakefs.paths]
    else:
        paths = cfg.lakefs.paths

    repo = cfg.lakefs.repository
    datapaths = {}

    for path in paths:
        # Find all case directories (identified by case_metadata.json)
        full_path = os.path.join(cfg.lakefs.branch, path)
        pages = paginator.paginate(Bucket=repo, Prefix=full_path)

        case_dirs = set()

        for page in pages:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                key = obj['Key']

                if is_metadata_file(key):
                    # Extract case directory
                    case_dir = os.path.dirname(key)
                    case_dirs.add(case_dir)

        print(f"Found {len(case_dirs)} potential cases in {path}")

        # Process each case directory
        for case_dir in sorted(case_dirs):
            case_id = os.path.basename(case_dir)

            case_data = extract_case_data(s3, repo, case_dir)

            if case_data:
                # Use case_id as the key (could be filename or rater/case combination)
                if case_id in datapaths:
                    print(f"Warning: Duplicate case_id {case_id}, will be overwritten")

                datapaths[case_id] = case_data
                print(
                    f"  ✓ Loaded case: {case_id} ({len(case_data['labels'])} slices, bounds: {case_data['bounds']['z_coords']})")
            else:
                print(f"  ✗ Skipped incomplete case: {case_id}")

    print(f"\nTotal valid cases loaded: {len(datapaths)}")

    # Only keep cases with complete data
    datapaths = {
        case_id: data
        for case_id, data in datapaths.items()
        if "image" in data and len(data["labels"]) > 0 and "bounds" in data
    }

    return datapaths


def create_data_split(cfg):
    """
    Stores paths to image files and label files in json files, split into train, val & test
    """

    datapaths = get_data_paths(cfg)

    if len(datapaths) == 0:
        raise ValueError("No valid cases found! Check your data paths and S3 configuration.")

    # Shuffle and split into training, validation and test data
    ids = list(datapaths.keys())
    random.Random(cfg.seeds.datasplit).shuffle(ids)

    if cfg.data.split.train + cfg.data.split.val + cfg.data.split.test != 100:
        raise ValueError(
            f"Data split ({cfg.data.split.train}, {cfg.data.split.val}, {cfg.data.split.test}) "
            f"provided in config should sum up to 100."
        )

    num_total = len(ids)
    num_train = int(round(num_total * cfg.data.split.train / 100))
    num_val = int(round(num_total * cfg.data.split.val / 100))

    train = [datapaths[id] for id in ids[:num_train]]
    val = [datapaths[id] for id in ids[num_train:num_train + num_val]]
    test = [datapaths[id] for id in ids[num_train + num_val:]]

    print(f"\nData split:")
    print(f"  Training:   {len(train)} cases")
    print(f"  Validation: {len(val)} cases")
    print(f"  Test:       {len(test)} cases")

    # Store data split
    if not os.path.exists(cfg.data.data_split_path):
        os.makedirs(cfg.data.data_split_path, exist_ok=True)

    train_path = os.path.join(cfg.data.data_split_path, "train.json")
    val_path = os.path.join(cfg.data.data_split_path, "val.json")
    test_path = os.path.join(cfg.data.data_split_path, "test.json")

    with open(train_path, "w") as f:
        json.dump(train, f, indent=4)
    with open(val_path, "w") as f:
        json.dump(val, f, indent=4)
    with open(test_path, "w") as f:
        json.dump(test, f, indent=4)

    print(f"\nSaved data splits to: {cfg.data.data_split_path}")


def main():
    cli_conf = OmegaConf.from_cli()
    if cli_conf.get('config') is not None:
        cfg_conf = OmegaConf.load(cli_conf.config)
        cfg = OmegaConf.merge(cfg_conf, cli_conf)
    else:
        cfg = cli_conf

    # Validate required config fields
    required_fields = [
        'lakefs.s3_endpoint',
        'lakefs.repository',
        'lakefs.branch',
        'lakefs.paths',
        'data.data_split_path',
        'data.split.train',
        'data.split.val',
        'data.split.test',
        'seeds.datasplit'
    ]

    for field in required_fields:
        if OmegaConf.select(cfg, field) is None:
            raise ValueError(f"Required configuration field '{field}' is missing")

    # Create train-val-test split
    create_data_split(cfg)


if __name__ == "__main__":
    main()