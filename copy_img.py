import os  # os是用来切换路径和创建文件夹的。
import shutil
import pandas as pd

#TODO:改成read poses, key frame
sequence=["00","02","03","04","05","06","07","08","09","10"]
for element in sequence:
    pose_file_path = r'/mnt/KITTI-360/data_poses/2013_05_28_drive_00'+element+'_sync/poses.txt'  # pose 的存放路径
    save_path = r'/mnt/KITTI-360/image_2/'+element  # save_path 是想把复制出来的文件存放在的路径
    image_path=r"/mnt/hdd_0/KITTI-360/data_2d_raw/2013_05_28_drive_00"+element+"_sync/image_00/data_rect" # 下载好的image路径
    pose = pd.read_csv(pose_file_path, sep=" ", index_col=False)        # print(newfile_path)
    key_frame=pose.values[:,0]
    dict={}
    for num in key_frame:
        if(element=="09"):
            if(int(num)>=9885 and int(num)<=10215):continue
        dict[int(num)]=1
    print(dict)
    ind=0
    os.makedirs(save_path)
    for filename in sorted(os.listdir(image_path)):  # 遍历pathDir下的所有文件filename
        if (int(filename[:-4]) not in dict): continue
        from_path = os.path.join(image_path, filename)  # 旧文件的绝对路径(包含文件的后缀名)
        to_path = save_path  # 新文件的绝对路径
        newfile_path = os.path.join(save_path, str(ind)+"_image.png")
        ind+=1
        shutil.copyfile(from_path, newfile_path)

