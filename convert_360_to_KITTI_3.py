import numpy as np
from plyfile import PlyData
from collections import namedtuple
from xml.dom import minidom
import re
import os
import glob

frame_dict = {}
def parseMatrix(data, rows, cols):  # 分割matrix
    if (type(data) != type([])):
        data = data.split(' ')
    mat = []
    for d in data:
        d = d.replace('\n', '')
        if len(d) < 1:
            continue
        mat.append(float(d))
    mat = np.reshape(mat, [rows, cols])
    return mat

def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    #inv_Tr[3,3]=1
    #print(inv_Tr)
    return inv_Tr




def readcam2world(path, word2cam=False):  # 读取cam0_to_world
    index=0
    with open(file=path, mode="r") as f:
        data = f.readlines()
        ret = {}
        index+=1
        for d in data:
            frame = int(d.split()[0])
            frame_dict[index]=frame
            index+=1
            if word2cam:
                ret[frame] = np.linalg.inv(parseMatrix(d.split()[1:], 4, 4))
            else:
                ret[frame] = parseMatrix(d.split()[1:], 4, 4)
        return ret


Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'kittiId',  # An integer ID that is associated with this label for KITTI-360
    # NOT FOR RELEASING

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'ignoreInInst',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations of instance segmentation or not

    'color',  # The color of this label
])
"""
my_dict={0:0,1:40,3:48,2:44,10:16,11:50,7:52,8:51,30:49,31:49,32:49,21:80, 23:99,24:81,5:70,4:72,
         9:99, 19:30,20:31,13:10,14:18,34:13,16:20,15:20,33:20,17:15,18:11,12:52,35:52,36:20,
         29:99,22:99,25:99,26:99,27:99,28:99,39:99,37:99}"""
my_dict={0:0,6:49,7:40,8:48,9:44,10:16,11:50,12:52,13:51,14:52,15:52,16:52,17:80,18:80, 19:81,20:81,21:70,22:72,
         23:0, 24:30,25:31,26:10,27:18,28:13,29:20,30:20,31:20,32:15,33:11,34:50,35:52,36:52,37:80,38:99,39:99,40:99,41:99,42:52,43:20,
         44:99}

labels = [
    #       name                     id    kittiId,    trainId   category            catId     hasInstances   ignoreInEval   ignoreInInst   color
    Label('unlabeled', 0, -1, 255, 'void', 0, False, True, True, (0, 0, 0)),
    Label('ego vehicle', 1, -1, 255, 'void', 0, False, True, True, (0, 0, 0)),
    Label('rectification border', 2, -1, 255, 'void', 0, False, True, True, (0, 0, 0)),
    Label('out of roi', 3, -1, 255, 'void', 0, False, True, True, (0, 0, 0)),
    Label('static', 4, -1, 255, 'void', 0, False, True, True, (0, 0, 0)),
    Label('dynamic', 5, -1, 255, 'void', 0, False, True, True, (111, 74, 0)),
    Label('ground', 6, -1, 255, 'void', 0, False, True, True, (81, 0, 81)),
    Label('road', 7, 1, 0, 'flat', 1, False, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 3, 1, 'flat', 1, False, False, False, (244, 35, 232)),
    Label('parking', 9, 2, 255, 'flat', 1, False, True, True, (250, 170, 160)),
    Label('rail track', 10, 10, 255, 'flat', 1, False, True, True, (230, 150, 140)),
    Label('building', 11, 11, 2, 'construction', 2, True, False, False, (70, 70, 70)),
    Label('wall', 12, 7, 3, 'construction', 2, False, False, False, (102, 102, 156)),
    Label('fence', 13, 8, 4, 'construction', 2, False, False, False, (190, 153, 153)),
    Label('guard rail', 14, 30, 255, 'construction', 2, False, True, True, (180, 165, 180)),
    Label('bridge', 15, 31, 255, 'construction', 2, False, True, True, (150, 100, 100)),
    Label('tunnel', 16, 32, 255, 'construction', 2, False, True, True, (150, 120, 90)),
    Label('pole', 17, 21, 5, 'object', 3, True, False, True, (153, 153, 153)),
    Label('polegroup', 18, -1, 255, 'object', 3, False, True, True, (153, 153, 153)),
    Label('traffic light', 19, 23, 6, 'object', 3, True, False, True, (250, 170, 30)),
    Label('traffic sign', 20, 24, 7, 'object', 3, True, False, True, (220, 220, 0)),
    Label('vegetation', 21, 5, 8, 'nature', 4, False, False, False, (107, 142, 35)),
    Label('terrain', 22, 4, 9, 'nature', 4, False, False, False, (152, 251, 152)),
    Label('sky', 23, 9, 10, 'sky', 5, False, False, False, (70, 130, 180)),
    Label('person', 24, 19, 11, 'human', 6, True, False, False, (220, 20, 60)),
    Label('rider', 25, 20, 12, 'human', 6, True, False, False, (255, 0, 0)),
    Label('car', 26, 13, 13, 'vehicle', 7, True, False, False, (0, 0, 142)),
    Label('truck', 27, 14, 14, 'vehicle', 7, True, False, False, (0, 0, 70)),
    Label('bus', 28, 34, 15, 'vehicle', 7, True, False, False, (0, 60, 100)),
    Label('caravan', 29, 16, 255, 'vehicle', 7, True, True, True, (0, 0, 90)),
    Label('trailer', 30, 15, 255, 'vehicle', 7, True, True, True, (0, 0, 110)),
    Label('train', 31, 33, 16, 'vehicle', 7, True, False, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 17, 'vehicle', 7, True, False, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 18, 'vehicle', 7, True, False, False, (119, 11, 32)),
    Label('garage', 34, 12, 2, 'construction', 2, True, True, True, (64, 128, 128)),
    Label('gate', 35, 6, 4, 'construction', 2, False, True, True, (190, 153, 153)),
    Label('stop', 36, 29, 255, 'construction', 2, True, True, True, (150, 120, 90)),
    Label('smallpole', 37, 22, 5, 'object', 3, True, True, True, (153, 153, 153)),
    Label('lamp', 38, 25, 255, 'object', 3, True, True, True, (0, 64, 64)),
    Label('trash bin', 39, 26, 255, 'object', 3, True, True, True, (0, 128, 192)),
    Label('vending machine', 40, 27, 255, 'object', 3, True, True, True, (128, 64, 0)),
    Label('box', 41, 28, 255, 'object', 3, True, True, True, (64, 64, 128)),
    Label('unknown construction', 42, 35, 255, 'void', 0, False, True, True, (102, 0, 0)),
    Label('unknown vehicle', 43, 36, 255, 'void', 0, False, True, True, (51, 0, 51)),
    Label('unknown object', 44, 37, 255, 'void', 0, False, True, True, (32, 32, 32)),
    Label('license plate', -1, -1, -1, 'vehicle', 7, False, True, True, (0, 0, 142)),
]

# --------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
# --------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label = {label.name: label for label in labels}
# id to label object
id2label = {label.id: label for label in labels}
# trainId to label object
trainId2label = {label.trainId: label for label in reversed(labels)}
# KITTI-360 ID to cityscapes ID
kittiId2label = {label.kittiId: label for label in labels}

kitti360_root = "/home/allenzj/KITTI-360"
img_width = 256*2
img_height = 256*2


def transform_pt(pt, pt_to_world, ref_to_world):
    # transform from world to local 1
    pt = np.dot(np.linalg.inv(pt_to_world), pt)

    # transform from local 2 to world
    pt = np.dot(ref_to_world, pt)
    return pt


def local2global(semanticId, instanceId):
    globalId = semanticId * 1000 + instanceId
    if isinstance(globalId, np.ndarray):
        return globalId.astype(np.int)
    else:
        return int(globalId)


def global2local(globalId):
    semanticId = globalId // 1000
    instanceId = globalId % 1000
    if isinstance(globalId, np.ndarray):
        return semanticId.astype(np.int), instanceId.astype(np.int)
    else:
        return int(semanticId), int(instanceId)


def get_toworld_matrix(obj):
    # original data from file is in string form with elements like '\n' and ''
    str_matrix = obj.getElementsByTagName("transform")[0].getElementsByTagName("data")[0].firstChild.data.split(" ")

    # convert values into float form using eval()
    float_matrix = []
    for each in str_matrix:
        if each != '' and each != '\n':
            float_matrix.append(eval(each))

    # reshape the list of floats into the right shape as a transformation matrix, then append to trans_matrices
    return np.reshape(float_matrix, (4, 4))


def main():
    save_dir="/home/allenzj/Semantic-KITTI/sequences/06"
    label_folder = os.path.join(save_dir, 'labels')
    calib_folder = save_dir
    lidar_folder = os.path.join(save_dir, 'velodyne')
    pose_folder = save_dir
    calib_folder = save_dir

    frame_dict={}

    for folder in [label_folder, calib_folder, lidar_folder, pose_folder]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    pcd_dir = kitti360_root + "/data_3d_semantics/train/2013_05_28_drive_0003_sync/dynamic/*"
    dynamic_lst=sorted(glob.glob(pcd_dir))
    static_lst=sorted(glob.glob(kitti360_root + "/data_3d_semantics/train/2013_05_28_drive_0003_sync/static/*"))
    #pcd_dir = kitti360_root + "/data_3d_semantics/train/2013_05_28_drive_0000_sync/dynamic/0000009886_0000010098.ply"
    box_dir = kitti360_root + "/data_3d_bboxes/train/2013_05_28_drive_0003_sync.xml"
# not sure
    poses_file = "/home/allenzj/KITTI-360/data_poses/2013_05_28_drive_0003_sync/cam0_to_world.txt"
    #poses_file = open("/home/allenzj/KITTI-360/data_poses/2013_05_28_drive_0000_sync/cam0_to_world.txt", "r")
    calib_file=open("/home/allenzj/KITTI-360/calibration/calib_cam_to_velo.txt","r")

    calib_lines = calib_file.readlines()
    for line in calib_lines:
        line = re.split('\n| ', line)
        matrix = []
        for each in line:
            if each != '':
                matrix.append(eval(each))
        matrix+=[0,0,0,1]
        cam_to_velo= np.reshape(matrix, (4, 4))
        #print(poses[tp])
    calib_file.close()
    """
    poses_lines = poses_file.readlines()
    poses = {}
    for line in poses_lines:
        line = re.split('\n| ', line)
        tp = int(line[0])
        matrix = []
        for each in line[1:]:
            if each != '':
                matrix.append(eval(each))
        matrix+=[0,0,0,1]
        poses[tp] = np.reshape(matrix, (4, 4))
        #print(poses[tp])
    poses_file.close()

    calib_lines = calib_file_1.readlines()
    for line in calib_lines:
        line = re.split('\n| ', line)
        matrix = []
        for each in line[1:]:
            if each != '':
                matrix.append(eval(each))
        matrix+=[0,0,0,1]
        GPS_to_camera= np.linalg.inv(np.reshape(matrix, (4, 4)))
        break
        #print(poses[tp])
    calib_file_1.close()
    """

    calib_path = os.path.join(calib_folder, 'calib.txt')
    calib=inverse_rigid_trans(cam_to_velo)
    with open(calib_path, "w") as calib_file:
        key="Tr"
        val = calib.flatten()[:12]
        val_str = '%.12e' % val[0]
        for v in val[1:]:
            val_str += ' %.12e' % v
        calib_file.write('%s: %s\n' % (key, val_str))


    poses = readcam2world(poses_file)
    """
    filename = os.path.join(save_dir, 'poses.txt')
    with open(filename, 'w') as f:
        for v in poses.values():
            for i in range(3):
                for j in range(4):
                    f.write(str(v[i, j]))
                    f.write(" ")

            f.write("\n")
    """
    #raise  Exception()
    frame_2_dict={}
    labels_lst={}
    velodyne_lst={}

    box = minidom.parse(box_dir)
    objs = box.getElementsByTagName("opencv_storage").item(0).childNodes
    for each in objs:
        if each.nodeValue == '\n':
            objs.remove(each)
    dynamic_objs = []
    for each in objs:
        # print(each.getElementsByTagName("timestamp")[0].firstChild.data)
        if each.getElementsByTagName("timestamp")[0].firstChild.data != "-1":
            dynamic_objs.append(each)
    #print(dynamic_objs)


    reference_to_world = {}

    dynamic_ins = []
    trans_matrices = {}
    dict={}
    # get the transformation matrices from bbox record file
    for i in range(len(dynamic_objs)):
        # the dynamic object
        obj = dynamic_objs[i]
        # instance ID of the object
        insId = int(obj.getElementsByTagName("instanceId")[0].firstChild.data)
        #print(insId)
        # dynamic_ins.append(int(local2global(semId, insId)))
        dynamic_ins.append(insId)

        # timestamp
        tp = int(obj.getElementsByTagName("timestamp")[0].firstChild.data)

        float_matrix = get_toworld_matrix(obj)
        trans_matrices[(insId, tp)] = float_matrix

        # save tuple of (instance ID, to-world transformation matrix)
        #if tp == int(target_timestamp):
        if (tp not in reference_to_world):
            reference_to_world[tp]={}
            reference_to_world[tp][insId] = float_matrix
        else:
            reference_to_world[tp][insId] = float_matrix
    #print(reference_to_world.keys())

    ind = 0
    for index in range(len(dynamic_lst)):
        plydata = PlyData.read(dynamic_lst[index])
        num_verts = plydata['vertex'].count
        np_ply_1 = np.zeros(shape=[num_verts, 11])
        # all points in global coordinate
        np_ply_1[:, 0] = plydata['vertex'].data['x']
        np_ply_1[:, 1] = plydata['vertex'].data['y']
        np_ply_1[:, 2] = plydata['vertex'].data['z']
        np_ply_1[:, 3] = plydata['vertex'].data['red']
        np_ply_1[:, 4] = plydata['vertex'].data['green']
        np_ply_1[:, 5] = plydata['vertex'].data['blue']
        np_ply_1[:, 6] = plydata['vertex'].data['semantic']
        np_ply_1[:, 7] = plydata['vertex'].data['instance']
        np_ply_1[:, 8] = plydata['vertex'].data['visible']
        np_ply_1[:, 9] = plydata['vertex'].data['timestamp']
        np_ply_1[:, 10] = plydata['vertex'].data['confidence']
        instance_id = np.array(np_ply_1[:, 7])
        timestamp = np.array(np_ply_1[:, 9])
        if(num_verts!=0):
            np_ply_1[:, 6] = np.vectorize(my_dict.get)(np_ply_1[:, 6].astype("int32"))

#len(np_ply_1[:, 0])
        for i in range(len(np_ply_1[:, 0])):
            insId = global2local(int(instance_id[i]))[1]
            tp = int(timestamp[i])
            if tp in reference_to_world and insId in reference_to_world[tp]:
                pt = np.append(np_ply_1[i][:3], 1)

                pt_to_world = trans_matrices[(insId, tp)]
                ref_to_world = reference_to_world[tp][insId]
                #print(1)
                # if tp in poses.keys():
                #     pos_to_world = poses[tp]
                # else:
                #     if tp > list(poses)[-1]:
                #         print("invalid data")
                #         pos_to_world = []
                #     else:
                #         while True:
                #             new_tp = tp + 1
                #             if new_tp in poses.keys():
                #                 pos_to_world = poses[new_tp]
                #                 break
                # pt = transform_pt(pt, pt_to_world, ref_to_world, pos_to_world)
                pt = transform_pt(pt, pt_to_world, ref_to_world)
                if(tp not in velodyne_lst):
                    velodyne_lst[tp]=pt.reshape(1,4)
                    labels_lst[tp] = [[np_ply_1[:, 6][i]]]
                    if([np_ply_1[:, 6][i]][0] not in dict):
                        dict[[np_ply_1[:, 6][i]][0]]=1
                    else:
                        dict[[np_ply_1[:, 6][i]][0]]+=1
                else:
                    velodyne_lst[tp] = np.r_[velodyne_lst[tp], pt.reshape(1, 4)]
                    labels_lst[tp] = np.append(labels_lst[tp], np_ply_1[:, 6][i])
                    if ([np_ply_1[:, 6][i]][0] not in dict):
                        dict[[np_ply_1[:, 6][i]][0]] = 1
                    else:
                        dict[[np_ply_1[:, 6][i]][0]] += 1

        plydata = PlyData.read(static_lst[index])
        num_verts = plydata['vertex'].count
        np_ply = np.zeros(shape=[num_verts, 11])
        # all points in global coordinate
        np_ply[:, 0] = plydata['vertex'].data['x']
        np_ply[:, 1] = plydata['vertex'].data['y']
        np_ply[:, 2] = plydata['vertex'].data['z']
        np_ply[:, 3] = plydata['vertex'].data['red']
        np_ply[:, 4] = plydata['vertex'].data['green']
        np_ply[:, 5] = plydata['vertex'].data['blue']
        np_ply[:, 6] = plydata['vertex'].data['semantic']
        np_ply[:, 7] = plydata['vertex'].data['instance']
        np_ply[:, 8] = plydata['vertex'].data['visible']
        #np_ply[:, 9] = plydata['vertex'].data['timestamp']
        np_ply[:, 10] = plydata['vertex'].data['confidence']
        frame_total=static_lst[index][:-4].split("/")[-1]
        start,end=frame_total.split("_")
        #TODO 怎么处理重复的
        if(index-1>-1):
            start=int(static_lst[index-1][:-4].split("/")[-1].split("_")[-1])+1
        for frame in range(int(start),int(end)+1):
            frame_2_dict[frame]=1
        #for frame in range(1031):
            if (frame not in poses): continue
            print(frame)
            curr_label=np.vectorize(my_dict.get)(np_ply[:, 6].astype("int32"))
            transform=inverse_rigid_trans(poses[frame]).astype("float32").T
            #transform=np.linalg.inv(poses[frame]).T.astype("float32")
            #print(poses[frame],frame)
           # print(transform)
            curr_pcl= np.c_[np_ply[:, 0].astype("float32").T,np_ply[:,1].astype("float32").T]
            curr_pcl= np.c_[curr_pcl,np_ply[:,2].astype("float32").T]
            curr_pcl=np.c_[curr_pcl,np.ones(curr_pcl.shape[0]).astype("float32").T]
            if(frame in velodyne_lst):
                curr_pcl=np.r_[curr_pcl,velodyne_lst[frame]]
                curr_label=np.append(curr_label,labels_lst[frame])
                del velodyne_lst[frame]
                del labels_lst[frame]
            new_pcl=curr_pcl@transform@calib.astype("float32")
            #new_pcl=curr_pcl@transform@inverse_rigid_trans(poses[frame][:3, :]).astype("int32")
            #new_pcl=curr_pcl@transform@(cam_to_velo.astype("float32"))
            """
            new_pcl = np.ones(curr_pcl.shape)
            new_pcl[:,0]=curr_pcl[:,2]
            new_pcl[:,1]=curr_pcl[:,0]*(-1)
            new_pcl[:, 2] = (-1)*curr_pcl[:, 1]
            """


            dst_lid_path = os.path.join(lidar_folder, '%06d' % (ind) + '.bin')
            with open(dst_lid_path, "w") as lid_file:
                new_pcl.astype("float32").tofile(lid_file)
            dst_label_path = os.path.join(label_folder, '%06d' % (ind) + '.label')
            with open(dst_label_path, "w") as label_file:
                curr_label.astype("int32").tofile(label_file)
            ind+=1
    #print(dict)

    filename = os.path.join(save_dir, 'poses.txt')
    index=0
    with open(filename, 'w') as f:
        frame=frame_dict[index]
        if(frame not in frame_2_dict):
            index+=1
            continue
        for v in poses.values():
            for i in range(3):
                for j in range(4):
                    f.write(str(v[i, j]))
                    f.write(" ")

            f.write("\n")
        index+=1
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

