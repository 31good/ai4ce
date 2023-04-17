## Visualize Voxels

The data is organized in the following format:

```
/kitti/dataset/
          └── sequences/
                  ├── 00/
                  │   ├── poses.txt
                  │   ├── image_2/
                  │   ├── image_3/
                  │   ├── pred/
                  │   │     ├ 000000.pred
                  │   │     └ 000001.pred
                  |   ├── voxels/
                  |   |     ├ 000000.bin
                  |   |     ├ 000000.label
                  |   |     ├ 000000.occluded
                  |   |     ├ 000000.invalid
                  |   |     ├ 000001.bin
                  |   |     ├ 000001.label
                  |   |     ├ 000001.occluded
                  |   |     ├ 000001.invalid
                  │   
                  │       
                  │       
                  ├── 01/
                  ├── 02/
                  .
                  .
                  .
                  └── 21/
```
Command: 
```sh
$ ./visualize_voxels.py --sequence 00 --dataset /path/to/kitti/dataset/
```
where:
- `sequence` is the sequence to be accessed.
- `dataset` is the path to the kitti dataset where the `sequences` directory is.

Navigation:
- `n` is next scan,
- `b` is previous scan,
- `esc` or `q` exits.

Note: Holding the forward/backward buttons triggers the playback mode.

In the Control Panel:
- `car` , `road` ... are class names for individual visualization
- `predict` and `ground-truth` work as switch only when classes selected it will show corresponding visualization
- `label` is for general ground truth.
- `predict_label` is for general prediction.


The main configuration file for the data is in `config/semantic-kitti.yaml`. `Auxillary` is for window building and the materials(shaders) for visualization.

Update: all moving objects like moving car and moving person are removed from the ground truth visualization

## Convert Nuscenes to KITTI Datasets
INPUT:
```
/Nuscenes/dataset/
          └── FILENAME/
                  ├── lidaseg/   
                  │   ├── V1.0(version_name)/
                  ├── maps/
                  ├── smaples/
                  ├── sweeps/
                  └── V1.0(version_name)/
```
Command: 
```sh
$ python3.8 convert_Nuscence_KITTI.py nuscenes_gt_to_kitti --nusc_kitti_dir /path/to/output/ --dataroot /path/to/Nuscence_dataset --image_count: number of frame --nusc_version: /filename_in_lidarseg(version_name)
```
OUTPUT:
```
/dataroot/
          └── val_(imagecount)_(camera)/
                  ├── ImageSets/   
                  ├── label_2/
                  ├── labels/
                  ├── velodyne/
                  ├── calib.txt
                  └── poses.txt
```
Note: Voxels need to extract information from labels, using the save_voxels.exe below.

## Save_voxels:
INPUT:
```
/paht_to_input/
        ├── labels/
        ├── velodyne/
        ├── calib.txt
        └── poses.txt
```
Command: 
```sh
$ ./save_voxels --indir=/path/to/input --outdir=/path/to/output --num=number_of_future_frame_to_fuse
```
OUTPUT:
```
/path_output/
          └── sequence_name/
                  |     ├ 000000.bin
                  |     ├ 000000.label
                  |     ├ 000000.occluded
                  |     ├ 000000.invalid
                  |     ├ 000001.bin
                  |     ├ 000001.label
                  |     ├ 000001.occluded
                  |     ├ 000001.invalid
```
NOTE: dependency of this execution and code need to be implemented. 

Chek https://github.com/jbehley/voxelizer/tree/0ace074b8981ded4c9bb73ad362b44ae2e57ad34.

Code is modified from this voxelizer.

## Procedure
Use converter first to convert Nuscences to KITTI format, then use save_voxels to get the voxels for n future frames fused data after converting. The result could be viewed by the visulaizer 
