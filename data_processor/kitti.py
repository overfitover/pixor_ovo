from __future__ import division
import os
import os.path
import torch
import numpy as np
import cv2
import math
from kitti_tools import *
from torch.utils.data import Dataset, DataLoader

class KittiDataset(torch.utils.data.Dataset):
    geometry = {
        'L1': -40.0,
        'L2': 40.0,
        'W1': 0.0,
        'W2': 70.0,
        'H1': -2.5,
        'H2': 1.0,
        'input_shape': (800, 700, 39),
        'label_shape': (200, 175, 7)
    }

    def __init__(self, root='/home/yxk/data/Kitti/object', set='train', type='velodyne_train'):
        self.type = type
        self.root = root
        self.data_path = os.path.join(root, 'training')
        self.lidar_path = os.path.join(self.data_path, "velo_new_bin/")
        self.image_path = os.path.join(self.data_path, "image_2/")
        self.calib_path = os.path.join(self.data_path, "calib/")
        self.label_path = os.path.join(self.data_path, "label_2/")

        with open(os.path.join(self.data_path, '%s.txt' % set)) as f:
            self.file_list = f.read().splitlines()

    def __getitem__(self, i):

        lidar_file = self.lidar_path + '/' + self.file_list[i] + '.bin'
        calib_file = self.calib_path + '/' + self.file_list[i] + '.txt'
        label_file = self.label_path + '/' + self.file_list[i] + '.txt'
        image_file = self.image_path + '/' + self.file_list[i] + '.png'
        #print(self.file_list[i])

        if self.type == 'velodyne_train':
            calib = load_kitti_calib(calib_file)
            target, _ = get_target(label_file, calib['Tr_velo2cam'], calib['R0'])
            target = torch.from_numpy(target)

            pointcloud = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 7)
            scan = self.lidar_preprocess(pointcloud)      # make feature map  (800, 700, 36)
            scan = torch.from_numpy(scan)
            return scan, target

        elif self.type == 'velodyne_test':
            NotImplemented

        else:
            raise ValueError('the type invalid')

    def __len__(self):
        return len(self.file_list)
    def get_label(self, i):
        calib_file = self.calib_path + '/' + self.file_list[i] + '.txt'
        label_file = self.label_path + '/' + self.file_list[i] + '.txt'
        calib = load_kitti_calib(calib_file)
        label_map, label_list = get_target(label_file, calib['Tr_velo2cam'], calib['R0'])
        return label_map, label_list


    def point_in_roi(self, point):
        if (point[0] - self.geometry['W1']) < 0.01 or (self.geometry['W2'] - point[0]) < 0.01:
            return False
        if (point[1] - self.geometry['L1']) < 0.01 or (self.geometry['L2'] - point[1]) < 0.01:
            return False
        if (point[2] - self.geometry['H1']) < 0.01 or (self.geometry['H2'] - point[2]) < 0.01:
            return False
        return True

    def lidar_preprocess(self, scan):  
        """
        precess point cloud, make feature map
        """
        velo = scan
        velo_processed = np.zeros(self.geometry['input_shape'], dtype=np.float32)
        intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))
        for i in range(velo.shape[0]):
            if self.point_in_roi(velo[i, :]):
                x = int((velo[i, 1]-self.geometry['L1']) / 0.1)
                y = int((velo[i, 0]-self.geometry['W1']) / 0.1)
                z = int((velo[i, 2]-self.geometry['H1']) / 0.1)
                velo_processed[x, y, z] = 1
                velo_processed[x, y, 35] += velo[i, 3]
                velo_processed[x, y,36] += velo[i, 4]
                velo_processed[x, y,37] += velo[i, 5]
                velo_processed[x, y,38] += velo[i, 6]

                intensity_map_count[x, y] += 1
        velo_processed[:, :, 35] = np.divide(velo_processed[:, :, 35],  intensity_map_count, \
                        where=intensity_map_count!=0)
        velo_processed[:, :, 36] = np.divide(velo_processed[:, :, 36],  intensity_map_count, \
                        where=intensity_map_count!=0)
        velo_processed[:, :, 37] = np.divide(velo_processed[:, :, 37],  intensity_map_count, \
                        where=intensity_map_count!=0)
        velo_processed[:, :, 38] = np.divide(velo_processed[:, :, 38],  intensity_map_count, \
                        where=intensity_map_count!=0)
        return velo_processed

def get_data_loader(batch_size):
    train_dataset = KittiDataset(root='/home/ovo/data/data/Kitti/object')
    train_data_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
    val_dataset = KittiDataset(root='/home/ovo/data/data/Kitti/object')
    val_data_loader = DataLoader(val_dataset, batch_size=1)

    return train_data_loader, val_data_loader

if __name__ == "__main__":
    a = KittiDataset(root='/home/ovo/data/data/Kitti/object', set='train')
    scan, target = a.__getitem__(1)
    print(np.shape(scan))
    print(np.shape(target))
