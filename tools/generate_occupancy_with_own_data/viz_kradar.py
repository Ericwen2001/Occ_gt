import os, sys
import cv2, imageio
import numpy as np
import torch
import open3d as o3d
import time
from open3d.open3d.geometry import voxel_down_sample,estimate_normals


def box_center_to_corner(box):
   
    translation = box[0:3]
    l, w, h = box[3], box[4], box[5]
    rotation = box[6]

    # Create a bounding box outline
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [0,0,0,0, h, h, h, h]])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(
        rotation_matrix, bounding_box) + eight_points.transpose()

    return corner_box.transpose()


def corner_box_to_line_set(corner_box):

    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
            [4, 5], [5, 6], [6, 7], [4, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [[1, 0, 0] for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corner_box)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set


def main():

    colors = np.array(
        [
            [205, 209, 228, 255], #static
            [100, 200, 255, 255],  # sedan              blue
            [255, 255, 100, 255],  # bus or truck              yellow
            [250, 55, 71, 255],  # motorcylce                  red
           [230, 230, 250, 255],  # bicyle   white
           [245, 40, 145, 255],  # pedestrian          pink
            [244, 158, 238, 255],  # pedestrian   group  light pink      
            [255, 240, 150, 255],  # bicycle group         light yellow
            [135, 100, 0, 255],  # unkown              brown
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
    pc_range = [-50, -50, -5, 50, 50, 5]
    root_path = "/mnt/data/DataSet/K-RadarOOC/"
    save_img = False
    read_view = True
    save_view = False
    
    # process one clip once
    splits = ['train']
    for split in splits:
        # clips = sorted(os.listdir(os.path.join(root_path, split)), key=lambda x:int(x.split("_")[1]))
        for clip in ['1']:
            # if not clip == 'delft_13':
            #     continue 
            print('***Start to process {}***'.format(clip))
            path = os.path.join(root_path, split, clip)
            occ_path = os.path.join(path,'occupancy_gt/')
            box_path = os.path.join(path,'bbox/')
            vis_path = os.path.join(path, 'occ_vis/')
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)
            len_sequence = np.minimum(250, len(os.listdir(occ_path)))
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1280,height=720)
            pcd = o3d.geometry.PointCloud()
            for i in range(len_sequence):
                fov_voxels = np.load(os.path.join(occ_path, 'occupancy_gt{}.npy'.format(i)))
                bboxes = np.load(os.path.join(box_path, 'bbox{}.npy'.format(i)))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(fov_voxels)
                # voxelGrid = o3d.geometry.VoxelGrid().create_from_point_cloud(pcd)
                voxelGrid = voxel_down_sample(pcd, voxel_size=voxel_size)
                # voxelGrid.paint_uniform_color([0.5, 0.5, 0.5])
                vis.add_geometry(voxelGrid)
                line_sets = []
                for j in range(bboxes.shape[0]):
                    corner_box = box_center_to_corner(bboxes[j])
                    line_set = corner_box_to_line_set(corner_box)
                    line_sets.append(line_set)
                for line_set in line_sets:
                    vis.add_geometry(line_set)
                ctr = vis.get_view_control()
                if read_view:
                    param = o3d.io.read_pinhole_camera_parameters('/home/xiangyu/SurroundOcc/tools/generate_occupancy_with_own_data/origin.json')
                    ctr.convert_from_pinhole_camera_parameters(param)       
                vis.poll_events()
                vis.update_renderer()
                vis.run()
                # time.sleep(0.05)
                # vis.capture_screen_image(os.path.join(images_outdir, "{}.png".format(file_name)))
                if save_img:
                    fname = os.path.join(vis_path, str(i).zfill(9) + '.png')
                    vis.capture_screen_image(fname)
                if save_view:
                    param = ctr.convert_to_pinhole_camera_parameters()
                    o3d.io.write_pinhole_camera_parameters('/home/xiangyu/SurroundOcc/tools/generate_occupancy_with_own_data/origin.json', param)
                    break
                vis.remove_geometry(voxelGrid)
                for line_set in line_sets:
                    vis.remove_geometry(line_set)
            vis.destroy_window()





if __name__ == '__main__':
    main()
