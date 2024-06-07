import os, sys
import cv2, imageio
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import time
import open3d as o3d
from open3d.open3d.geometry import voxel_down_sample,estimate_normals


def main():

    read_view = True
    save_view = False
    root_path = "/mnt/Kradar/K-Radar/"
    source_dirs = sorted(os.listdir(root_path))
    tgt_dir = "/mnt/data/DataSet/K-RadarOOC/"
    # process one clip once
    test_dirs = ['46','54'] #3 15 22 23 55
    for source in test_dirs:
        print('***Start to process sequence {}***'.format(source))
        vis_dir = os.path.join(tgt_dir, 'vis_lidar_pc_64', source)
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        file_dir = os.path.join(root_path, source, 'os2-64')
        pc_files = sorted(glob(file_dir + '/'+ '*.pcd'))
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1280,height=720)

        for pc_file in tqdm(pc_files):
            pcd = o3d.io.read_point_cloud(pc_file)
            # pcd_raw = np.fromfile(pc_file,dtype=np.float32)
            # num_points = pcd_raw.size // 4
            # lidar_points = pcd_raw.reshape((4,num_points))
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pcd_raw.points)  
            # pcd.colors = o3d.utility.Vector3dVector(pcd_raw.colors)
            vis.add_geometry(pcd)
            ctr = vis.get_view_control()
            if read_view:
                param = o3d.io.read_pinhole_camera_parameters('/home/xiangyu/SurroundOcc/tools/generate_occupancy_with_own_data/origin_lidar.json')
                ctr.convert_from_pinhole_camera_parameters(param)       
            vis.poll_events()
            vis.update_renderer()
            # vis.run()
            vis.capture_screen_image(os.path.join(vis_dir, 'lpc_{}.png'.format(pc_file.split('/')[-1].split('.')[0].split('_')[-1])))
            if save_view:
                param = ctr.convert_to_pinhole_camera_parameters()
                o3d.io.write_pinhole_camera_parameters('/home/xiangyu/SurroundOcc/tools/generate_occupancy_with_own_data/origin_lidar.json', param)
                break
            vis.remove_geometry(pcd)

        vis.destroy_window()
        del ctr
        del vis
        time.sleep(6)

if __name__ == '__main__':
    main()
