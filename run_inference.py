import torch
import time
import os
from Loss.loss import CustomLoss
from data_processor.kitti import get_data_loader
from Models.model1 import PIXOR
from utils import get_model_name, load_config, plot_bev, plot_label_map
from post_process.postprocess import non_max_suppression
import sys
import cv2
sys.path.insert(0, './')

# load dataset
config_name = 'config.json'
config, _, _, _ = load_config(config_name)
# set gpu
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# change this parameter to use specific devices
os.environ['CUDA_VISIBLE_DEVICES'] = str(
    config["use_gpu_id"]).strip('[').strip(']')  # "0, "


def inference():
    # evaluation
    config_name='config.json'
    config, _, _, _ = load_config(config_name)
    if torch.cuda.is_available():
        device = 'cuda'
        net = PIXOR(config['use_bn']).cuda()
    else:
        device = 'cpu'
        net = PIXOR(config['use_bn']).cpu()
    # net, criterion = build_model(config, device, train=False)
    
    net.load_state_dict(torch.load(get_model_name(config['name']), map_location=device))
    net.set_decode(True)
    # loader, _ = get_data_loader(batch_size=1, use_npy=config['use_npy'], frame_range=config['frame_range'])
    loader, _ = get_data_loader(batch_size=1)

    net.eval()
    # image_ids = [8, 3, 15, 27]
    image_id = 4
    threshold = config['cls_threshold']

    with torch.no_grad():
        # for image_id in image_ids:
        pc_feature, label_map = loader.dataset[image_id]
        pc_feature = pc_feature.to(device)
        label_map = label_map.to(device)
        label_map_unnorm, label_list = loader.dataset.get_label(image_id)
        
        # Forward Pass
        t_start = time.time()
        pred = net(pc_feature.unsqueeze(0)).squeeze_(0)
        print("Forward pass time", time.time() - t_start)

        # Select all the bounding boxes with classification score above threshold
        cls_pred = pred[..., 0]
        activation = cls_pred > threshold

        # Compute (x, y) of the corners of selected bounding box
        num_boxes = int(activation.sum())
        if num_boxes == 0:
            print("No bounding box found")
            return

        corners = torch.zeros((num_boxes, 8))
        for i in range(1, 9):
            corners[:, i - 1] = torch.masked_select(pred[..., i], activation)
        corners = corners.view(-1, 4, 2).numpy()
        scores = (torch.masked_select(pred[..., 0], activation)).cpu().numpy()

        # NMS
        t_start = time.time()
        # these are some problems. ????
        selected_ids = non_max_suppression(corners, scores, config['nms_iou_threshold'])
        corners = corners[selected_ids]
        scores = scores[selected_ids]
        print("Non max suppression time:", time.time() - t_start)

        # Visualization
        pc_feature = pc_feature.cpu().numpy()                        # (800, 700, 36)
        plot_bev(pc_feature, corners, label_list, window_name='predict_gt')
        # plot_bev(pc_feature, corners, window_name='Prediction')
        # plot_label_map(cls_pred.cpu().numpy())
        # cv2.waitKey (0)  
        

if __name__=='__main__':
    inference()
