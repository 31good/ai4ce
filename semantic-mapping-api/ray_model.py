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
from scipy.spatial.transform import Rotation   
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
parser.add_argument('--input', type=str, default='/mnt/NAS/home/zijun/test')
args = parser.parse_args()


def main_func(velodyne_path, poses_file, labels_path, out_dir, calib_tran, calib_tran_inv):
    # Initialize parameters
    _device = torch.device('cuda')
    voxel_size = [0.2, 0.2, 0.2]
    point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
    spatial_shape = [256, 256, 32]
    occlusion_theshold = 1

    # Read and process points, poses, and labels
    pcds = []
    labels = []
    origins = []
    pcd_files = sorted(os.listdir(velodyne_path))
    labels_files = sorted(os.listdir(labels_path))
    dynamic_point_files=sorted(os.listdir(dynamic_point))
    dynamic_label_points=sorted(os.listdir(dynamic_label))
    poses = np.loadtxt(poses_file, dtype=np.float32).reshape(-1, 3, 4)
    dummy = np.broadcast_to(np.array([0, 0, 0, 1]), (poses.shape[0], 1, 4))
    poses = np.concatenate([poses, dummy], axis=1)
    poses = calib_tran_inv @ poses @ calib_tran
    pose_first_inv = np.linalg.inv(poses[0])
    for j in range(len(pcd_files)//70,5):
        for i in range(j,j+70):
            points = np.fromfile(os.path.join(velodyne_path, pcd_files[i]), dtype=np.float32).reshape(-1, 4)[:, :3]
            label = np.fromfile(os.path.join(labels_path, labels_files[i]), dtype=np.int32)
            if(i==j and pcd_file[i] in dynamic_point_files):
                new_points=np.fromfile(os.paht.join(dynamic_point,pcd_files[i]),dytype=np.float32).reshape(-1,4)[:,:3]
                points=np.concatenate((points,new_points),axis=0)
                new_labels=np.fromfile(os.paht.join(dynamic_label_point,labels_files[i]),dtype=np.int32)
                label+=new_labels
                print(new_labels.shape,new_points.shape)
            pose = poses[i]
            print(pose)
            origin = np.concatenate([pose[:3, 3], np.array([1])], axis=0)

            points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
            points = points @ pose.T

            points = points @ pose_first_inv.T
            origin = origin @ pose_first_inv.T
            points = points[:, :3]
            origin = origin[:3]
            # print(origin)
            origin = np.broadcast_to(origin, (points.shape[0], 3))
            pcds.append(points)
            labels.append(label)
            origins.append(origin)
    # assert()
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

    final_voxel_label = voxel_label.cpu().numpy().astype(np.uint16)
    final_invalid = torch.logical_and(voxel_state == -1, voxel_label == 0).cpu().numpy().astype(np.uint8)
    final_voxel_label.tofile(os.path.join(out_dir, f"{str(index).zfill(6)}.label"))
    final_invalid.tofile(os.path.join(out_dir, f"{str(index).zfill(6)}.invalid"))



if __name__ == "__main__": 
    data_dir = args.input
    out_dir = os.path.join(data_dir, "sequences", "00", "voxels")
    velodyne_dir = os.path.join(data_dir, "velodyne")
    poses_file = os.path.join(data_dir, "poses.txt")
    labels_dir = os.path.join(data_dir, "labels")
    calib_file = os.path.join(data_dir, "calib.txt")
    dynamic_point=os.path.join("/mnt/NAS/home/zijun/dynamic","velodyne")
    dynamic_label=os.path.join("/mnt/NAS/home/zijun/dynamic","labels")
    with open (calib_file, "r") as f:
        calib = f.readlines()[5]
    calib_tran = np.fromstring(calib.split(": ")[-1], sep=" ", dtype=np.float32).reshape(3, 4)
    calib_tran = np.concatenate([calib_tran, np.array([[0, 0, 0, 1]])], axis=0)
    calib_tran_inv = np.linalg.inv(calib_tran)
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)

    """
    velodyne_folders = sorted(os.listdir(velodyne_dir))
    #poses_files = sorted(os.listdir(poses_dir))
    labels_folders = sorted(os.listdir(labels_dir))
    """

    main_func(velodyne_dir,poses_file,labels_dir,out_dir,calib_tran,calib_tran_inv,dynamic_label,dynamic_point)
    """
    for i in tqdm(range(len(velodyne_folders))):
        velodyne_path = os.path.join(velodyne_dir, velodyne_folders[i])
        poses_file = os.path.join(poses_dir, poses_files[i])
        labels_path = os.path.join(labels_dir, labels_folders[i])
        main_func(velodyne_path, poses_file, labels_path, out_dir, calib_tran, calib_tran_inv, i)
    """
