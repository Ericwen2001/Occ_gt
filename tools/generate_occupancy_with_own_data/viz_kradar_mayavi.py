import os, sys
import cv2, imageio
import numpy as np
import torch
import time
import mayavi.mlab as mlab



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
        [   [0, 0, 0, 255], # unoccupied  white
            [205, 209, 228, 255], #static
            [100, 200, 255, 255],  # sedan              blue
            [255, 255, 100, 255],  # bus or truck              yellow
            [250, 55, 71, 255],  # motorcylce                  red
           [230, 230, 250, 255],  # bicyle   white
           [245, 40, 145, 255],  # pedestrian          pink
            [244, 158, 238, 255],  # pedestrian   group  light pink      
            [255, 240, 150, 255],  # bicycle group         light yellow
            [135, 100, 0, 255],  # unkown              brown
        ]
    ).astype(np.uint8)

    colors_nus = np.array([
        [0  , 0  , 0, 255],
        [100, 150, 245, 255],
        [100, 230, 245, 255],
        [30, 60, 150, 255],
        [80, 30, 180, 255],
        [100, 80, 250, 255],
        [255, 30, 30, 255],
        [255, 40, 200, 255],
        [150, 30, 90, 255],
        [255, 0, 255, 255],
        [255, 150, 255, 255],
        [75, 0, 75, 255],
        [175, 0, 75, 255],
        [255, 200, 0, 255],
        [255, 120, 50, 255],
        [0, 175, 0, 255],
        [135, 60, 0, 255],
        [150, 240, 80, 255],
        [255, 240, 150, 255],
        [255, 0, 0, 255]]) / 255.0

    voxel_size = 0.95
    pc_range = [-50, -50, -5, 50, 50, 5]
    root_path = "/mnt/data/DataSet/K-RadarOOC/"
    save_img = True
    read_view = True
    save_view = False
    
    # process one clip once
    splits = ['train']
    for split in splits:
        clips = sorted(os.listdir(os.path.join(root_path, split)), key=lambda x:int(x))
        for clip in clips:
            if not clip == '1':
                continue 
            print('***Start to process {}***'.format(clip))
            path = os.path.join(root_path, split, clip)
            occ_path = os.path.join(path,'semantic_occupancy_gt/')
            if not os.path.exists(occ_path):
                continue
            box_path = os.path.join(path,'bbox/')
            vis_path = os.path.join(path, 'occ_vis_mlab_fov/')
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)
            len_sequence = np.minimum(900, len(os.listdir(occ_path)))

            # for i in range(len_sequence):
            for i in range(len_sequence):
                fov_voxels = np.load(os.path.join(occ_path, 'occupancy_gt_with_semantic{}.npy'.format(i)))
                fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
                # pdb.set_trace()
                # fov_voxels = fov_voxels[fov_voxels[:,0]>99]
                print(np.unique(fov_voxels[:, 3]))

                plt_plot_fov = mlab.points3d(
                    fov_voxels[:, 0],
                    fov_voxels[:, 1],
                    fov_voxels[:, 2],
                    fov_voxels[:, 3],
                    colormap="cool",
                    scale_factor=voxel_size,
                    mode="cube",
                    opacity=1.0,
                    vmax = 8,
                    vmin= 0
                )

                plt_plot_fov.glyph.scale_mode = "scale_by_vector"
                plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
                # mlab.view(180, 60, 600)
                # mlab.yaw(45)



                # mlab.show()
                mlab.savefig(os.path.join(vis_path, str(i).zfill(9) + '.png'))
                mlab.clf(figure = fig)
                mlab.close()





if __name__ == '__main__':
    main()
