import os  # os是用来切换路径和创建文件夹的。
import shutil

labels_file_path = r'/mnt/hdd_0/KITTI-360/sequences/03/single/labels'  # 想拆分的文件夹所在路径,也就是一大堆文件所在的路径
save_path = r'/mnt/hdd_0/KITTI-360/sequences/03/single/velodyne'  # save_dir 是想把复制出来的文件存放在的路径
velodyne_path=r"/mnt/hdd_0/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data_rect"
for filename in os.listdir(labels_file_path):
    dict[filename] = 1
for filename in velodyne_path:  # 遍历pathDir下的所有文件filename
    # if(filename[-5]=="e"):continue
    # if(int(filename[:-4]) not in index):continue
    if (filename[-10:-4] + ".label" not in dict): continue
    # print(filename)
    from_path = os.path.join(velodyne_path, filename)  # 旧文件的绝对路径(包含文件的后缀名)
    to_path = save_path  # 新文件的绝对路径
    newfile_path = os.path.join(save_path, filename)
    # print(newfile_path)
    shutil.copyfile(from_path, newfile_path)

