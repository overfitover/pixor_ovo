'''
Load pointcloud/labels from the KITTI dataset folder
'''
import os.path
import numpy as np
import time
import torch
from utils import plot_bev, get_points_in_a_rotated_box, plot_label_map, trasform_label2metric
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#KITTI_PATH = '/home/autoronto/Kitti/object'
KITTI_PATH = 'KITTI'

class KITTI(Dataset):

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

    target_mean = np.array([0.008, 0.001, 0.202, 0.2, 0.43, 1.368])
    target_std_dev = np.array([0.866, 0.5, 0.954, 0.668, 0.09, 0.111])

    def __init__(self, frame_range = 10000, use_npy=False, train=True):
        self.frame_range = frame_range
        self.velo = []
        self.use_npy = use_npy

        self.image_sets = self.load_imageset(train) # names
        self.transform = transforms.Normalize(self.target_mean, self.target_std_dev)
        self.load_velo()

    def __len__(self):
        return len(self.image_sets)

    def __getitem__(self, item):
        scan = self.load_velo_scan(item)            # (-1, 4)
        if not self.use_npy:
            scan = self.lidar_preprocess(scan)      # make feature map  (800, 700, 36)
        scan = torch.from_numpy(scan)
        label_map, _ = self.get_label(item)
        self.reg_target_transform(label_map)
        label_map = torch.from_numpy(label_map)
        return scan, label_map

    def reg_target_transform(self, label_map):
        '''
        Inputs are numpy arrays (not tensors!)
        :param label_map: [200 * 175 * 7] label tensor
        :return: normalized regression map for all non_zero classification locations
        '''
        cls_map = label_map[..., 0]
        reg_map = label_map[..., 1:]

        index = np.nonzero(cls_map)
        reg_map[index] = (reg_map[index] - self.target_mean)/self.target_std_dev

    def load_imageset(self, train):
        path = KITTI_PATH
        if train:
            path = os.path.join(path, "train.txt")
        else:
            path = os.path.join(path, "val.txt")

        with open(path, 'r') as f:
            lines = f.readlines() # get rid of \n symbol
            names = []
            for line in lines[:-1]:
                if int(line[:-1]) < self.frame_range:      # filter out of frame_range
                    names.append(line[:-1])

            # Last line does not have a \n symbol
            names.append(lines[-1][:6])
            # print(names[-1])
            print("There are {} images in txt file".format(len(names)))

            return names
    
    def box3d_cam_to_velo(self, box3d, Tr, R0):
        """
        :param box3d: camera  w,h,l,x,y,z,ry
        :param Tr:
        :param R0: 
        :return: lidar
        """   
        def project_cam2velo(cam, Tr, R0):
            """
            :param cam: (4, 1)   tx, ty, tz, 1
            :param Tr: (4, 4)
            :param R0: (4, 4)
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

        h, w, l, tx, ty, tz, ry = [float(i) for i in box3d]
        cam = np.ones([4, 1])
        cam[0] = tx
        cam[1] = ty
        cam[2] = tz
        t_lidar = project_cam2velo(cam, Tr, R0)
        
        return t_lidar

    def get_corners(self, bbox):
        '''
        camera coordinate to lidar coordinate
        why change w h location  ??
        '''

        h, w, l, y, z, x, yaw = bbox[8:15]
        y = -y   # make y face right

        # h, w, l, x, y, z, yaw = bbox[8:15]

        # manually take a negative s. t. it's a right-hand system, with
        # x facing in the front windshield of the car
        # z facing up
        # y facing to the left of driver

        yaw = -(yaw + np.pi / 2)    # make yaw face forward
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        # rear left
        bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        # rear right
        bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front right
        bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front left
        bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]

        return bev_corners, reg_target


    def update_label_map(self, map, bev_corners, reg_target):
        '''
        x forward y right --> x right y forward
        '''
        label_corners = (bev_corners / 4 ) / 0.1                       # 4x downsamples 
        label_corners[:, 1] += self.geometry['label_shape'][0] / 2     # 200, 175, 7   ?????

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
 
    def get_label(self, index):
        '''
        :param i: the ith velodyne scan in the train/val set
        :return: label map: <--- This is the learning target
                a tensor of shape 800 * 700 * 7 representing the expected output


                label_list: <--- Intended for evaluation metrics & visualization
                a list of length n; n =  number of cars + (truck+van+tram+dontcare) in the frame
                each entry is another list, where the first element of this list indicates if the object
                is a car or one of the 'dontcare' (truck,van,etc) object

        '''
        index = self.image_sets[index]
        f_name = (6-len(index)) * '0' + index + '.txt'
        label_path = os.path.join(KITTI_PATH, 'training', 'label_2', f_name)

        object_list = {'Car': 1,'Truck':0, 'DontCare':0, 'Van':0, 'Tram':0}
        label_map = np.zeros(self.geometry['label_shape'], dtype=np.float32)
        label_list = []
        with open(label_path, 'r') as f:
            lines = f.readlines() # get rid of \n symbol
            for line in lines:
                bbox = []
                entry = line.split(' ')
                name = entry[0]
                if name in list(object_list.keys()):
                    bbox.append(object_list[name])
                    bbox.extend([float(e) for e in entry[1:]])
                    if name == 'Car':
                        corners, reg_target = self.get_corners(bbox)     # get corner(4, 2) and (cos, sin, x, y, w, l)
                        self.update_label_map(label_map, corners, reg_target)
                        label_list.append(corners)
        return label_map, label_list

    def get_rand_velo(self):
        import random
        rand_v = random.choice(self.velo)
        print("A Velodyne Scan has shape ", rand_v.shape)
        return random.choice(self.velo)

    def load_velo_scan(self, item):
        """Helper method to parse velodyne binary files into a list of scans."""
        filename = self.velo[item]

        if self.use_npy:
            scan = np.load(filename[:-4]+'.npy')
        else:
            scan = np.fromfile(filename, dtype=np.float32).reshape(-1, 7)

        return scan

    def load_velo(self):
        """Load velodyne [x,y,z,reflectance] scan data from binary files."""
        # Find all the Velodyne files
        velo_files = []
        for file in self.image_sets:
            file = '{}.bin'.format(file)
            velo_files.append(os.path.join(KITTI_PATH, 'training', 'velo_new_bin', file))

        print('Found ' + str(len(velo_files)) + ' Velodyne scans...')
        # Read the Velodyne scans. Each point is [x,y,z,reflectance]
        self.velo = velo_files

        print('done.')

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


def get_data_loader(batch_size, use_npy=False, frame_range=10000):
    train_dataset = KITTI(frame_range, use_npy=use_npy,train=True)
    # train_dataset.load_velo()
    train_data_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
    
    val_dataset = KITTI(frame_range, use_npy=use_npy, train=False)
    # val_dataset.load_velo()
    val_data_loader = DataLoader(val_dataset, batch_size=1)

    return train_data_loader, val_data_loader


def test0():
    k = KITTI()
    id = 4
    k.load_velo()
    tstart = time.time()
    scan = k.load_velo_scan(id)
    processed_v = k.lidar_preprocess(scan)
    print(np.shape(processed_v))

    label_map, label_list = k.get_label(id)
    print(np.shape(label_map))
    print('time taken: %gs' %(time.time()-tstart))
    plot_bev(processed_v, label_list=label_list)
    plot_label_map(label_map[:, :, 0])


def find_reg_target_var_and_mean():
    k = KITTI()
    k.load_velo()
    reg_targets = [[] for _ in range(6)]
    for i in range(len(k)):
        label_map, _ = k.get_label(i)
        car_locs = np.where(label_map[:, :, 0] == 1)
        for j in range(1, 7):
            map = label_map[:, :, j]
            reg_targets[j-1].extend(list(map[car_locs]))

    reg_targets = np.array(reg_targets)
    means = reg_targets.mean(axis=1)
    stds = reg_targets.std(axis=1)

    np.set_printoptions(precision=3, suppress=True)
    print("Means", means)
    print("Stds", stds)
    return means, stds

def preprocess_to_npy(train=True):
    k = KITTI(train=train)
    k.load_velo()
    for item, name in enumerate(k.velo):
        scan = k.load_velo_scan(item)
        scan = k.lidar_preprocess(scan)
        path = name[:-4] + '.npy'
        np.save(path, scan)
        print('Saved ', path)
    return

def test():
    train_data_loader, val_data_loader = get_data_loader(2)
    for i, (input, label_map) in enumerate(train_data_loader):
        print("Entry", i)
        print("Input shape:", input.shape)
        print("Label Map shape", label_map.shape)
        if i == 2:
            break

    print("Finish testing train dataloader")


if __name__=="__main__":
    test0()
    # preprocess_to_npy(True)
    # preprocess_to_npy(False)
    
