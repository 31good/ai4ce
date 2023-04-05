from accumulation import *
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--kitti_dir", help="path to kitti360 dataset")
parser.add_argument("-o", "--output_dir", help="path to output_dir")
args = parser.parse_args()
root_dir = args.kitti_dir
output_dir = args.output_dir

root_dir="/home/allenzj/KITTI-360"
output_dir="/home/allenzj/Semantic-KITTI/sequences/10/single"
velodyne_file_path=r'/home/allenzj/KITTI-360/data_3d_raw/2013_05_28_drive_0003_sync/velodyne_points/data'
labels_file_path=r'/home/allenzj/Semantic-KITTI/sequences/03/labels'
dict={}
for sequence in os.listdir(os.path.join(root_dir,"data_3d_raw")):
    all_spcds = os.listdir(os.path.join(os.path.join(os.path.join(root_dir,"data_3d_semantics/train"),sequence),"static"))
    all_spcds.sort()
    for i in range(len(all_spcds)):
        spcd = all_spcds[i]
        if i == 0:
            spcd_prev = None 
        else:
            spcd_prev = all_spcds[i-1]
        if i == len(all_spcds)-1:
            spcd_next = None 
        else:
            spcd_next = all_spcds[i+1]
        partial_name = os.path.splitext(spcd)[0].split('_')
        first_frame = int(partial_name[0])
        last_frame = int(partial_name[1])
        print(first_frame,last_frame)
        # travel_padding=20 (origin)
        PA = PointAccumulation(root_dir, output_dir, sequence, first_frame, last_frame, 51.2, 1, 0.02, True, True)
        #PA.createOutputDir()
        PA.loadTransformation()
        PA.getInterestedWindow()
        PA.loadTimestamps()
        PA.addVelodynePoints()
        PA.getPointsInRange()
        PA.recoverLabel(spcd,spcd_prev,spcd_next)
        """
        PA.writeToFiles()
        mkdir(output_dir+"/velodyne")
        save_path = output_dir+"/velodyne"
        pathDir = sorted(os.listdir(velodyne_file_path))  # os.listdir(file_path) 是获取指定路径下包含的文件或文件夹列表
        for filename in os.listdir(labels_file_path):
            dict[filename] = 1
        for filename in pathDir:  # 遍历pathDir下的所有文件filename
            # if(filename[-5]=="e"):continue
            # if(int(filename[:-4]) not in index):continue
            if (filename[:-4] + ".label" not in dict): continue
            #print(filename)
            from_path = os.path.join(velodyne_file_path, filename)  # 旧文件的绝对路径(包含文件的后缀名)
            to_path = save_path  # 新文件的绝对路径
            newfile_path = os.path.join(save_path, filename)
            # print(newfile_path)
            shutil.copyfile(from_path, newfile_path)
        """
#shutil.copyfile(src, dst)
#shutil.copyfile(src, dst)
