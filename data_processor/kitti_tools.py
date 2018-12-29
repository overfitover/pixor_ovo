import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import cv2
import math

import scipy.misc as misc
from utils import plot_bev, get_points_in_a_rotated_box, plot_label_map, trasform_label2metric


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

def update_label_map(map, bev_corners, reg_target):
        '''
        x forward y right --> x right y forward
        '''
        label_corners = (bev_corners / 4 ) / 0.1                       # 4x downsamples 
        label_corners[:, 1] += geometry['label_shape'][0] / 2     # 200, 175, 7   ?????

        points = get_points_in_a_rotated_box(label_corners)

        for p in points:
            label_x = p[0]
            label_y = p[1]
            metric_x, metric_y = trasform_label2metric(np.array(p)) 
            actual_reg_target = np.copy(reg_target)                  # cos, sin, x, y, w,l
            actual_reg_target[2] = reg_target[2] - metric_x          # dx
            actual_reg_target[3] = reg_target[3] - metric_y          # dy
            actual_reg_target[4] = np.log(reg_target[4])             # log(w)
            actual_reg_target[5] = np.log(reg_target[5])             # log(l)

            map[label_y, label_x, 0] = 1.0
            map[label_y, label_x, 1:7] = actual_reg_target           # x to right  y to forward

def get_target(label_file, Tr, R0):
    """
    :param label_file: 
    :param Tr: 
    :param R0: 
    :return: 
    """
    
    object_list = {'Car': 1,'Truck':0, 'DontCare':0, 'Van':0, 'Tram':0}
    label_map = np.zeros(geometry['label_shape'], dtype=np.float32)
    label_list = []
    with open(label_file, 'r') as f:
        lines = f.readlines() # get rid of \n symbol
        for line in lines:
            bbox = []
            entry = line.split(' ')
            name = entry[0]
            if name in list(object_list.keys()):
                bbox.append(object_list[name])
                bbox.extend([float(e) for e in entry[1:]])
                if name == 'Car':
                    # print("bbox: ",bbox[8:], "Tr: ", Tr,"R0: ", R0)
                    corners, reg_target = box3d_cam_to_velo(bbox[8:], Tr, R0)     # get corner(4, 2) and (cos, sin, x, y, w, l)
                    update_label_map(label_map, corners, reg_target)
                    label_list.append(corners)
    return label_map, label_list
    

def box3d_cam_to_velo(box3d, Tr, R0):
    """
    :param box3d: camera  w,h,l,x,y,z,ry
    :param Tr: 
    :param R0: 
    :return: 
    """

    def project_cam2velo(cam, Tr, R0):
        """
        :param cam: (4, 1)   tx, ty, tz, 1
        :param Tr: (3, 4)
        :param R0: (3, 3)
        :return: (3, 1)
        """
        R_cam_to_rect = np.eye(4)
        R_cam_to_rect[:3, :3] = np.array(R0).reshape(3, 3)
        cam = np.matmul(np.linalg.inv(R_cam_to_rect), cam)
        T = np.zeros([4, 4], dtype=np.float32)
        T[:3, :] = Tr
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)

    def ry_to_rz(ry):
        angle = -ry - np.pi / 2
        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2*np.pi + angle
        return angle

    h, w, l, tx, ty, tz, ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr, R0)
    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0, 0, 0, 0, h, h, h, h]])
    rz = ry_to_rz(ry)
    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])
    velo_box = np.dot(rotMat, Box)                            # (3, 8)
    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T
    box3d_corner = cornerPosInVelo.transpose()                # (8, 3)
    box3d_corner = box3d_corner[:4, :2]

    # print(rz)
    reg_target = [np.cos(rz), np.sin(rz), t_lidar[0][0], t_lidar[0][1], w, l]
    return box3d_corner.astype(np.float32), reg_target

def load_kitti_calib(calib_file):

    """
    load projection matrix
    """
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)

    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

if __name__ == '__main__':
    import numpy as np
    # box = np.random.randn(8, 3)
    # box = box[:4,:2]
    # print(box.shape)
    box = [1, 2, 3,4,5, 6, 7]
    Tr = np.random.randn(3, 4)
    R0 = np.random.randn(3, 3)
    box3d_cam_to_velo(box, Tr, R0)



