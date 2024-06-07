import os
import cv2
import yaml
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from vod.common import get_frame_list
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader
from vod.frame import FrameTransformMatrix
from vod.frame import homogeneous_transformation
from vod.frame import FrameLabels
from scipy.spatial.transform import Rotation as R
from PIL import Image


MAP_TABLE = {
            "Unoccupied": 0,
            "Static": 1,
            "Car": 2,
            "Pedestrian": 3,
            "Cyclist": 4,
            "rider": 5,
            "bicycle": 6,
            "bicycle_rack": 7,
            "human_depiction": 8,
            "moped_scooter": 9,
            "motor": 10,
            "truck": 11,
            "ride_other": 12,
            "vehicle_other": 13,
            "ride_uncertain": 14,
            "DontCare": 15
}
MAP_TABLE = {
            "Unoccupied": 0,
            "Static": 1,
            "Car": 2,
            "Pedestrian": 2,
            "Cyclist": 2,
            "rider": 2,
            "bicycle": 2,
            "bicycle_rack": 2,
            "human_depiction": 2,
            "moped_scooter": 2,
            "motor": 2,
            "truck": 2,
            "ride_other": 2,
            "vehicle_other": 2,
            "ride_uncertain": 2,
            "DontCare": 2
}
def main():

    root_dir = "/mnt/data/fangqiang/view_of_delft/"
    clip_dir =  "/home/xiangyu/SurroundOcc/tools/generate_occupancy_with_own_data/clips/"
    label_dir = '/home/xiangyu/SurroundOcc/tools/generate_occupancy_with_own_data/label_2_track/'
    output_dir = "/mnt/data/fangqiang/vod_occ_format/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open('/home/xiangyu/SurroundOcc/tools/generate_occupancy_with_own_data/clips_info.yaml','r') as f:
        splits = yaml.safe_load(f.read())
    data_loc = KittiLocations(root_dir=root_dir)

    for split in splits:
        for clip in splits[split]:
            frames = get_frame_list(clip_dir+'/'+clip+'.txt')
            print('{} - {}'.format(clip, len(frames)))
            if split == 'train' or split == 'val':
            # if split == 'val':
                save_path = os.path.join(output_dir, split, clip)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                vod_data_conversion(frames, data_loc, clip, save_path, label_dir)

def vod_data_conversion(frames, data_loc, clip, save_path, label_dir):
    
    num_frames = len(frames)
    lidar_save_path = os.path.join(save_path, 'pc')
    radar_save_path = os.path.join(save_path, 'rpc')

    pose_save_path = os.path.join(save_path, 'pose')
    cam_save_path = os.path.join(save_path, 'cam')
    bbox_save_path = os.path.join(save_path, 'bbox')
    if not os.path.exists(cam_save_path):
        os.makedirs(cam_save_path)
    if not os.path.exists(lidar_save_path):
        os.makedirs(lidar_save_path)
    if not os.path.exists(radar_save_path):
        os.makedirs(radar_save_path)
    if not os.path.exists(pose_save_path):
        os.makedirs(pose_save_path)
    if not os.path.exists(bbox_save_path):
        os.makedirs(bbox_save_path)
    

    for i in tqdm(range(len(frames)), desc=clip):
        frame = frames[i]
        data = FrameDataLoader(kitti_locations=data_loc,
                    frame_number=frame)
        camera_data = data.image
        image=Image.fromarray(camera_data)
        image.save(os.path.join(cam_save_path, '{}_cam_{}.png'.format(clip,i)))
        # # save lidar pc as .npy file
        # lidar_pc = data.lidar_data[:, :4]
        # lidar_pc = np.array(lidar_pc)
        # lidar_pc.tofile(os.path.join(lidar_save_path, 'pc{}.bin'.format(i)))
        # transforms = FrameTransformMatrix(data)
        # radar_pc = data.radar_data[:,:]
        # radar_pc = np.array(radar_pc)
        # radar_pc = transfrom_radar(radar_pc,transforms)
        # radar_pc.tofile(os.path.join(radar_save_path, 'rpc{}.bin'.format(i)))

        # np.save(os.path.join(lidar_save_path, 'pc{}.npy'.format(i)), lidar_pc)



        # # save lidar pose relative to the odom 
        # transforms = FrameTransformMatrix(data)
        # lidar_ego_pose = np.dot(transforms.t_odom_camera, transforms.t_camera_lidar)
        # np.save(os.path.join(pose_save_path, 'lidar_ego_pose{}.npy'.format(i)), lidar_ego_pose)

        # # save boxes info 
        # labels = load_track_labels(label_dir, frame)
        # bboxes = np.array(labels[0])
        # ids = np.array(labels[1])
        # types= pd.Series(labels[2])
        # types_map = types.map(MAP_TABLE).to_numpy()
        # if any(np.isnan(types_map)):
        #     print(types)
        #     raise Exception('NaN type names, please modify the mapping table')
        # new_bboxes = transform_labels(bboxes, transforms)
        # np.save(os.path.join(bbox_save_path, 'bbox{}.npy'.format(i)), new_bboxes)
        # np.save(os.path.join(bbox_save_path, 'bbox_token{}.npy'.format(i)), ids)
        # np.save(os.path.join(bbox_save_path, 'object_category{}.npy'.format(i)), types_map)

def transfrom_radar(radar_pc, transforms):
    new_radar_pc = np.zeros_like(radar_pc)
    for j in range(radar_pc.shape[0]):
        obj_info = radar_pc[j,:]
        center = (transforms.t_radar_lidar @ np.array([obj_info[0], obj_info[1], obj_info[2], 1]))[:3]

        new_radar_pc[j, :3] = center
        new_radar_pc[j, 3:] = radar_pc[j, 3:]
    return new_radar_pc


def transform_labels(labels, transforms):

    num_obj = labels.shape[0]
    new_labels = np.zeros((num_obj, 7))
    for j in range(num_obj):
        obj_info = labels[j,:]
        center = (transforms.t_lidar_camera @ np.array([obj_info[3], obj_info[4], obj_info[5], 1]))[:3]

        extent = np.array([obj_info[2], obj_info[1], obj_info[0]]) # l w h
        angle = - (obj_info[6] + np.pi / 2)
        new_labels[j, :3] = center
        new_labels[j, 3:6] = extent
        new_labels[j,6] = angle

    return new_labels
    

    
def load_track_labels(label_path, frame):

    label_file = label_path + '/' + frame + '.txt'
    if os.path.exists(label_file):
        with open(label_file, 'r') as text:
            labels = text.readlines()
        labels = get_track_labels(labels)
    else:
        labels = []
 
    return labels


def get_track_labels(labels):
    
    bboxes = []
    ids = []
    types = []
    for act_line in labels:  # Go line by line to split the keys
        act_line = act_line.split()
        if len(act_line)==17:
            label, id, _, _, _, _, _, _, _, h, w, l, x, y, z, rot, score = act_line
        if len(act_line)==16:
            label, id, _, _, _, _, _, _, h, w, l, x, y, z, rot, score = act_line
        h, w, l, x, y, z, rot, score = float(h), float(w), float(l), float(x), float(y), float(z), float(rot), float(score)
        # for debug 
        # h = 10
        bboxes.append([h, w, l, x, y, z, rot])
        ids.append(id)
        types.append(label)

    return [bboxes, ids, types]



if __name__ == "__main__":
    main()
