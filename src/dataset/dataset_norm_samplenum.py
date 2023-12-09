import os
import torch
from torch.utils.data import Dataset
import numpy as np
import json

import random
random.seed(20)




def generate_classname_classidx_mapping():
    """
    skip: 30 39 57 58
    """
    class_idx = 0
    mapping_dict = {}
    for i in range(60):
        if i in [30, 39, 57, 58]:
            continue
        mapping_dict[str(i)] = class_idx
        class_idx += 1
    return mapping_dict


def get_split_vname_label():
    mapping_dict = generate_classname_classidx_mapping()

    with open('./data/BABEL/3d_obj/kit.json', 'r') as json_f:

        json_obj = json.load(json_f)
    split_dict = {'train': [], 'val': [], 'test': []}
    vidname_label_dict = {}
    for _, ele in json_obj.items():
        for split_name, vidname_lst in ele.items():
            split_dict[split_name].extend(vidname_lst)

    for label_str, ele in json_obj.items():
        label_num = mapping_dict[label_str]
        for _, vidname_lst in ele.items():
            for vidname in vidname_lst:
                vidname_label_dict[vidname] = label_num

    return split_dict, vidname_label_dict


class NTU_RGBD_norm_samplenum(Dataset):
    """NTU depth human masked datasets"""

    def __init__(self, root_path, opt,
                 full_train=True,
                 test=False,
                 validation=False,
                 DATA_CROSS_VIEW=True,
                 Transform=True):

        self.DATA_CROSS_VIEW = DATA_CROSS_VIEW
        self.root_path = root_path
        self.SAMPLE_NUM = opt.SAMPLE_NUM
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM

        self.EACH_FRAME_SAMPLE_NUM = 128

        self.all_framenum = opt.all_framenum
        self.framenum = opt.framenum
        self.transform = Transform
        # self.depth_path = opt.depth_path

        self.splits_dict, self.vidname_label_dict = get_split_vname_label()

        self.point_vids = os.listdir(self.root_path)  # .sort()
        self.point_vids.sort()

        self.num_clouds = len(self.point_vids)
        print(self.num_clouds)

        self.train = (test == False) and (validation == False)

        if test:
            print('use test split')
            self.vid_ids = self.splits_dict['test'].copy()
        elif validation:
            print('use validation split')
            self.vid_ids = self.splits_dict['val'].copy()
        elif full_train:
            print('use train split')
            self.vid_ids = self.splits_dict['train'].copy()
        else:
            print('error: unknown split')
        print('num_data:', len(self.vid_ids))

    def __getitem__(self, idx):

        vid_id = self.vid_ids[idx]
        vid_name = vid_id

        path_T = self.root_path
        path_cloud_npy_T = os.path.join(path_T, vid_name + '.npy')
        geodist_npy = os.path.join(path_T, vid_name + '_geodist.npz')
        geodist = np.load(geodist_npy)['arr_0']

        frame_index = []
        for jj in range(self.framenum):
            iii = int(np.random.randint(
                int(self.all_framenum * jj / self.framenum),
                int(self.all_framenum * (jj + 1) / self.framenum)))
            frame_index.append(iii)

        points4DV_T = np.load(path_cloud_npy_T)
        max_x = points4DV_T[:, :, 0].max()
        max_y = points4DV_T[:, :, 1].max()
        max_z = points4DV_T[:, :, 2].max()
        min_x = points4DV_T[:, :, 0].min()
        min_y = points4DV_T[:, :, 1].min()
        min_z = points4DV_T[:, :, 2].min()

        x_len = max_x - min_x
        y_len = max_y - min_y
        z_len = max_z - min_z

        x_center = (max_x + min_x) / 2
        y_center = (max_y + min_y) / 2
        z_center = (max_z + min_z) / 2

        points4DV_T[:, :, 0] = (points4DV_T[
                                :, :,
                                0] - x_center) / y_len
        points4DV_T[:, :, 1] = (points4DV_T[
                                :, :,
                                1] - y_center) / y_len
        points4DV_T[:, :, 2] = (points4DV_T[
                                :, :,
                                2] - z_center) / y_len

        points4DV_T = points4DV_T[frame_index, 0:self.EACH_FRAME_SAMPLE_NUM,
                      :]  # 60*512*4

        geodist = geodist[frame_index, :, :]
        geodist = geodist[:, 0:self.EACH_FRAME_SAMPLE_NUM, :]
        geodist = geodist[:, :, 0:self.EACH_FRAME_SAMPLE_NUM]
        label = self.vidname_label_dict[vid_name]
        theta = np.random.rand() * 1.4 - 0.7

        if self.transform:
            points4DV_T = self.point_transform(points4DV_T, theta)
        points4DV_T = torch.tensor(points4DV_T, dtype=torch.float)
        geodist = torch.tensor(geodist, dtype=torch.float)
        label = torch.tensor(label)
        return points4DV_T,geodist, label, vid_name

    def __len__(self):
        return len(self.vid_ids)

    def point_transform(self, points_xyz, y):

        anglesX = (np.random.uniform() - 0.5) * (1 / 9) * np.pi
        R_y = np.array([[[np.cos(y), 0.0, np.sin(y)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(y), 0.0, np.cos(y)]]])
        R_x = np.array([[[1, 0, 0],
                         [0, np.cos(anglesX), -np.sin(anglesX)],
                         [0, np.sin(anglesX), np.cos(anglesX)]]])

        points_xyz[:, :, 0:3] = self.jitter_point_cloud(points_xyz[:, :, 0:3],
                                                        sigma=0.007,
                                                        clip=0.04)  #

        R = np.matmul(R_y, R_x)


        points_xyz[:, :, 0:3] = np.matmul(points_xyz[:, :, 0:3], R)

        return points_xyz

    def jitter_point_cloud(self, data, sigma=0.01, clip=0.05):
        """

        :param data: Nx3 array
        :return: jittered_data: Nx3 array
        """
        M, N, C = data.shape
        # print(np.random.randn(M, N, C))#
        jittered_data = np.clip(sigma * np.random.randn(M, N, C), -1 * clip,
                                clip).astype(np.float32)  #

        jittered_data = data + jittered_data

        return jittered_data

    def random_dropout_point_cloud(self, data):
        """
        :param data:  Nx3 array
        :return: dropout_data:  Nx3 array
        """
        M, N, C = data.shape  ##60*300*4
        dropout_ratio = 0.7 + np.random.random() / 2  # n
        # dropout_ratio = np.random.random() * p
        drop_idx = np.where(np.random.random(N) <= dropout_ratio)[0]
        dropout_data = np.zeros_like(data)
        if len(drop_idx) > 0:
            dropout_data[:, drop_idx, :] = data[:, drop_idx, :]


        return dropout_data
