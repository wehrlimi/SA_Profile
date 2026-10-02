import numpy as np
import napari

# load npz
data = np.load(r"C:/Users/michael/Documents/Data/Noel_inference_output/axial_heatmaps.npz")

# create viewer
viewer = napari.Viewer()

# add each array to napari
for name in data.files:
    arr = data[name]

    if arr.ndim in (2, 3):
        # images / volumes
        viewer.add_image(arr, name=name)
    elif arr.ndim == 4:
        # e.g. (C, Z, H, W) or (Z, H, W, C)
        viewer.add_image(arr, name=name)
    elif arr.ndim == 2 and arr.shape[1] in (2, 3):
        # point clouds
        viewer.add_points(arr, name=name, size=2)
    else:
        print(f"Skipping {name}, shape={arr.shape}")

napari.run()
