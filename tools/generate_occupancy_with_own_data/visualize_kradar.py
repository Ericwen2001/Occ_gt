import os, sys
import cv2, imageio
import numpy as np
import torch
import open3d as o3d
from open3d.open3d.geometry import voxel_down_sample,estimate_normals


def main():

    colors = np.array(
        [
            [0, 0, 0, 255],
            [255, 120, 50, 255],  # barrier              orangey
            [255, 192, 203, 255],  # bicycle              pink
            [255, 255, 0, 255],  # bus                  yellow
            [0, 150, 245, 255],  # car                  blue
            [0, 255, 255, 255],  # construction_vehicle cyan
            [200, 180, 0, 255],  # motorcycle           dark orange
            [255, 0, 0, 255],  # pedestrian           red
            [255, 240, 150, 255],  # traffic_cone         light yellow
            [135, 60, 0, 255],  # trailer              brown
            [160, 32, 240, 255],  # truck                purple
            [255, 0, 255, 255],  # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red
            [139, 137, 137, 255],
            [75, 0, 75, 255],  # sidewalk             dard purple
            [150, 240, 80, 255],  # terrain              light green
            [230, 230, 250, 255],  # manmade              white
            [0, 175, 0, 255],  # vegetation           green
            [0, 255, 127, 255],  # ego car              dark cyan
            [255, 99, 71, 255],
            [0, 191, 255, 255]
        ]
    ).astype(np.uint8)

    voxel_size = 0.5
    pc_range = [-50, -50, -5, 50, 50, 3]
    root_path = "/mnt/data/DataSet/K-RadarOOC/"

    
    # process one clip once
    splits = ['train']
    for split in splits:
        # clips = sorted(os.listdir(os.path.join(root_path, split)), key=lambda x:int(x.split("_")[1]))
        for clip in ['4']:
            # if not clip == 'delft_13':
            #     continue 
            print('***Start to process {}***'.format(clip))
            path = os.path.join(root_path, split, clip)
            occ_path = os.path.join(path,'occupancy_gt/')
            vis_path = os.path.join(path, 'occ_vis/')
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)
            len_sequence = np.minimum(200, len(os.listdir(occ_path)))
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1280,height=720)
            print(len_sequence)

            for i in range(95,len_sequence):
                fov_voxels = np.load(os.path.join(occ_path, 'occupancy_gt{}.npy'.format(i)))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(fov_voxels)
                voxelGrid = voxel_down_sample(pcd, voxel_size=voxel_size)
                vis.add_geometry(voxelGrid)
                ctr = vis.get_view_control()
                
                vis.poll_events()
                vis.update_renderer()
                vis.run()
                # vis.capture_screen_image(os.path.join(images_outdir, "{}.png".format(file_name)))

                vis.destroy_window()


if __name__ == '__main__':
    main()
