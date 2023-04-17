# Copyright 2022 tao.jiang
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Tuple
# import open3d as o3d

import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
from utils.ray_traversal import *

NOT_OBSERVED = -1
FREE = 0
OCCUPIED = 1
FREE_LABEL = 0
VIS = False
# IMPORTANT: not use tf32 which case pose error
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=10)
parser.add_argument('--input', type=str, default='/mnt/hdd_0/KITTI-360/sequences/03/single')
args = parser.parse_args()


def main_func(velodyne_path, poses_file, labels_path, out_dir, index):
    # Initialize parameters
    _device = torch.device('cuda')
    voxel_size = [0.1, 0.1, 0.2]
    point_cloud_range = [-80, -80, -5, 80, 80, 7.8]
    spatial_shape = [1600, 1600, 64]
    occlusion_theshold = 1

    # Read and process points, poses, and labels
    pcds = []
    labels = []
    origins = []
    pcd_files = sorted(os.listdir(velodyne_path))
    labels_files = sorted(os.listdir(labels_path))
    for i in range(len(pcd_files)):
        points = np.fromfile(os.path.join(velodyne_path, pcd_files[i]), dtype=np.float32).reshape(-1, 4)[:, :3]
        label = np.fromfile(os.path.join(labels_path, labels_files[i]), dtype=np.int32)
        origin = np.loadtxt(poses_file, dtype=np.float32).reshape(-1, 3, 4)[i, :3, 3]
        origin = np.broadcast_to(origin, (points.shape[0], 3))
        pcds.append(points)
        labels.append(label)
        origins.append(origin)
    pcds = np.concatenate(pcds, axis=0)
    labels = np.concatenate(labels, axis=0)
    origins = np.concatenate(origins, axis=0)

    # Convert to torch tensor 
    pcds_dev = torch.from_numpy(pcds).to(_device)
    labels_dev = torch.from_numpy(labels).long().to(_device)
    origins_dev = torch.from_numpy(origins).to(_device)

    voxel_coors, voxel_state, voxel_label, voxel_occ_count, voxel_free_count = ray_traversal(
        origins_dev, pcds_dev, labels_dev,
        point_cloud_range, voxel_size, spatial_shape,
    )

    if VIS:
        from utils import vis_occ
        voxel_show = (voxel_state == 1)
        vis = vis_occ.main(voxel_label.cpu(), voxel_show.cpu(), voxel_size=[1, 1, 1], vis=None, offset=[0, 0, 0])
        vis.run()
        del vis

    final_voxel_label = voxel_label.to(torch.uint8)
    final_voxel_show = (voxel_state == 1).to(torch.uint8)
    save_name = os.path.join(out_dir, f"{str(index).zfill(3)}.npz")
    np.savez_compressed(
        save_name,
        voxel_label = final_voxel_label.cpu().numpy(),
        voxel_show = final_voxel_show.cpu().numpy()
    )


if __name__ == "__main__": 
    data_dir = args.input
    out_dir = os.path.join(data_dir, "voxels")
    velodyne_dir = os.path.join(data_dir, "velodyne")
    #poses_dir = os.path.join(data_dir, "poses.txt")
    labels_dir = os.path.join(data_dir, "labels")
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)
    poses_file = os.path.join(data_dir, "poses.txt")
    main_func(velodyne_dir,poses_file,labels_dir,out_dir,0)
    """
    velodyne_folders = sorted(os.listdir(velodyne_dir))
    poses_file = os.path.join(data_dir,"poses.txt")
    labels_folders = sorted(os.listdir(labels_dir))

    for i in tqdm(range(len(velodyne_folders))):
        velodyne_path = os.path.join(velodyne_dir, velodyne_folders[i])
        labels_path = os.path.join(labels_dir, labels_folders[i])
        main_func(velodyne_path, poses_file, labels_path, out_dir, i)
    """