import os, sys
import cv2, imageio
import numpy as np
import torch
import time
from glob import glob
from tqdm import tqdm
import open3d as o3d
from open3d.open3d.geometry import voxel_down_sample,estimate_normals


def main():

    read_view = False
    save_view = True
    root_path = "/mnt/data/DataSet/K-RadarOOC/train/"
    source_dirs = sorted(os.listdir(root_path))
    tgt_dir = "/mnt/data/DataSet/K-RadarOOC/"
    # process one clip once
    for source in ['21', '38', '39', '42', '45']:
        print('***Start to process sequence {}***'.format(source))
        vis_dir = os.path.join(tgt_dir, 'vis_radar_pc', source)
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        file_dir = os.path.join(root_path, source, 'radar_pc')
        pc_files = sorted(glob(file_dir + '/'+ '*.npy'))
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1280,height=720)

        for pc_file in tqdm(pc_files):
            radar_pc = np.load(pc_file)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(radar_pc[:, :3])
            
            # voxelGrid = voxel_down_sample(pcd, voxel_size=voxel_size)
            vis.add_geometry(pcd)
            ctr = vis.get_view_control()
            if read_view:
                param = o3d.io.read_pinhole_camera_parameters('/home/xiangyu/SurroundOcc/tools/generate_occupancy_with_own_data/origin_radar.json')
                ctr.convert_from_pinhole_camera_parameters(param)       
            vis.poll_events()
            vis.update_renderer()
            vis.run()
            vis.capture_screen_image(os.path.join(vis_dir, 'rpc_{}.png'.format(pc_file.split('/')[-1].split('.')[0].split('_')[-1])))
            if save_view:
                param = ctr.convert_to_pinhole_camera_parameters()
                o3d.io.write_pinhole_camera_parameters('/home/xiangyu/SurroundOcc/tools/generate_occupancy_with_own_data/origin_radar.json', param)
                break
            vis.remove_geometry(pcd)


        vis.destroy_window()
        del ctr
        del vis
        time.sleep(6)

if __name__ == '__main__':
    main()
