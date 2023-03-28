import os  # os是用来切换路径和创建文件夹的。

sequence=["00","02","03","04","05","06","07","09","10"]
for element in sequence:
    #pose_file_path = r'/mnt/KITTI-360/data_poses/2013_05_28_drive_00'+element+'_sync/poses.txt'  # pose 的存放路径
    labels_file_path = r'/mnt/hdd_0/KITTI-360/sequences/'+element+'/single/velodyne'  # single/velodyne存放路径
    image_path=r"/mnt/hdd_0/KITTI-360/data_2d_raw/2013_05_28_drive_00"+element+"_sync/image_00/data_rect" # 下载好的image路径
    #pose = pd.read_csv(pose_file_path, sep=" ", index_col=False)
    #key_frame=pose.values[:,0]
    dict={}
    for filename in os.listdir(labels_file_path):
        dict[int(filename[:-4])]=1
    ind=0
    for filename in sorted(os.listdir(image_path)):  # 遍历pathDir下的所有文件filename
        if (int(filename[:-4]) not in dict):
            os.remove(filename)
        #os.rename(from_path,newfile_path)
        from_path = os.path.join(image_path, filename)  # 旧文件的绝对路径(包含文件的后缀名)
        newfile_path = os.path.join(image_path, str(ind)+"_image.png")
        ind+=1
        os.rename(from_path,newfile_path)
    break
