import numpy as np
import os

velodyne_path="/mnt/hdd_0/KITTI-360/sequences/07/single/velodyne"
label_path="/mnt/hdd_0/KITTI-360/sequences/07/single/labels"
label_2_path="/mnt/hdd_0/KITTI-360/sequences/07/single_2/labels"
velodyne_files=sorted(os.listdir(velodyne_path))
label_files=sorted(os.listdir(label_path))
os.mkdir("mnt/hdd_0/KITTI-360/sequences/07/single_nondynamic/velodyne")
os.mkdir("mnt/hdd_0/KITTI-360/sequences/07/single_nondynamic/labels")
for i in range(len(velodyne_files)):
    points = np.fromfile(os.path.join(velodyne_path, velodyne_files[i]), dtype=np.float32).reshape(-1,4)
    label = np.fromfile(os.path.join(label_path, label_files[i]), dtype=np.int32)

    label_2 = np.fromfile(os.path.join(label_2_path, label_files[i]), dtype=np.int32)

    match_indices = np.where(label == label_2)[0]

    # Extract matching elements from `points` and `label_2`
    points_output = points[match_indices]
    label_output = label_2[match_indices]

    points_output.astype("float32").tofile("/mnt/hdd_0/KITTI-360/sequences/07/single_nondynamic/velodyne/"+velodyne_files[i])
    label_output.astype("int32").tofile("/mnt/hdd_0/KITTI-360/sequences/07/single_nondynamic/labels/"+label_files[i])
