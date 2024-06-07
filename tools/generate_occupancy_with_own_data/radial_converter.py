import os
import cv2
import yaml
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R
from PIL import Image
from radial_dataset import RADIal
from torch.utils.data import Dataset, DataLoader, Subset


def match_seq_frames(dataset):

    # get the correponding frames for each seq
    seq_dict = {}
    unique_seqs = np.unique(dataset.labels[:,14])
    for i,seq in enumerate(unique_seqs):
        sample_ids = np.where(dataset.labels[:,14] == seq)[0]
        frame_ids = np.unique(dataset.labels[sample_ids,0])
        seq_dict[seq] = frame_ids
    return seq_dict

def transfrom_radar(radar_pc, transforms):
    new_radar_pc = np.zeros_like(radar_pc)
    for j in range(radar_pc.shape[0]):
        obj_info = radar_pc[j,:]
        center = (transforms.t_radar_lidar @ np.array([obj_info[0], obj_info[1], obj_info[2], 1]))[:3]

        new_radar_pc[j, :3] = center
        new_radar_pc[j, 3:] = radar_pc[j, 3:]
    return new_radar_pc


def main():

    root_dir = '/mnt/data/DataSet/RADIal/'
    output_dir = '/mnt/data/fangqiang/RADIal_occ_format/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset = RADIal(root_dir = root_dir ,difficult=True)
    seq_dict = match_seq_frames(dataset)
    seq_keys = list(seq_dict.keys())
    dict_index_to_keys = {s:i for i,s in enumerate(dataset.sample_keys)}
    # Load the camera calibration parameters
    cam_calib = np.load(os.path.join(root_dir,'camera_calib.npy'),allow_pickle=True).item()

    for seq in seq_keys:
        seq_indexes = seq_dict[seq]
        seq_ids = [dict_index_to_keys[k] for k in seq_indexes]
        subset = Subset(dataset, seq_ids)
        len_seq = len(seq_ids)
        save_path = os.path.join(output_dir, seq)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
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

        for i in tqdm(range(len_seq), desc = seq):
            data = subset.__getitem__(i)
            # sample_id = int(subset.sample_keys[i])
            # save camera image as .png file
            camera_data = data[0]
            image=Image.fromarray(camera_data)
            image.save(os.path.join(cam_save_path, '{}_cam_{}.png'.format(seq,i)))
            # save lidar pc as .npy file
            lidar_pc = data[3]
            np.save(os.path.join(lidar_save_path, 'pc{}.npy'.format(i)), lidar_pc)
            # save radar pc as .npy file
            radar_pc = data[2]
            radar_pc.tofile(os.path.join(radar_save_path, 'rpc{}.bin'.format(i)))
            # save bbox info in the radar coordinates (reannotated by echofusion)
            labels = data[5]
            bboxes = labels.tolist()
            ids = []
            types = []
            # bboxes = np.array(labels[0])
            # ids = np.array(labels[1])
            # types= pd.Series(labels[2])
            # types_map = types.map(MAP_TABLE).to_numpy()
            # if any(np.isnan(types_map)):
            #     print(types)
            #     raise Exception('NaN type names, please modify the mapping table')
            # new_bboxes = transform_labels(bboxes, transforms)
            np.save(os.path.join(bbox_save_path, 'bbox{}.npy'.format(i)), bboxes)
            # np.save(os.path.join(bbox_save_path, 'bbox_token{}.npy'.format(i)), ids)
            # np.save(os.path.join(bbox_save_path, 'object_category{}.npy'.format(i)), types_map)
            
                    







if __name__ == "__main__":
    main()



