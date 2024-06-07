import os, sys
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


    colors_vod = np.array(
        [[0  , 0  , 0, 255],
        [205, 209, 228, 255],  # gray
        [100, 200, 255, 255],     #         blue
        [255, 255, 100, 255],  #               yellow
        [250, 55, 71, 255],  #                 red
        [245, 40, 145, 255],  #           pink
        [244, 158, 238, 255],  #      light pink      
        [255, 240, 150, 255],  #          light yellow
        [135, 100, 0, 255],  #             brown
        [255, 40, 200, 255],
        [150, 30, 90, 255],
        [255, 0, 255, 255],
        [255, 150, 255, 255],
        [75, 0, 75, 255],
        [175, 0, 75, 255],
        [150, 240, 80, 255]]
            # "Static": 1,
            # "Car": 2,
            # "Pedestrian": 3,
            # "Cyclist": 4,
            # "rider": 5,
            # "bicycle": 6,
            # "bicycle_rack": 7,
            # "human_depiction": 8,
            # "moped_scooter": 9,
            # "motor": 10,
            # "truck": 11,
            # "ride_other": 12,
            # "vehicle_other": 13,
            # "ride_uncertain": 14,
            # "DontCare": 15
    ).astype(np.uint8)
  

    voxel_size = 0.95
    pc_range = [-50, -50, -5, 50, 50, 5]
    root_path = "/mnt/data/fangqiang/vod_occ_format/"
    save_img = True
    read_view = True
    save_view = False
    
    # process one clip once
    splits = ['val']
    for split in splits:
        clips = sorted(os.listdir(os.path.join(root_path, split)), key=lambda x:int(x.split("_")[1]))
        for clip in clips:
            # if not clip == 'delft_2':
            #     continue 
            print('***Start to process {}***'.format(clip))
            path = os.path.join(root_path, split, clip)
            occ_path = os.path.join(path,'occupancy_gt_with_semantic/')
            box_path = os.path.join(path,'bbox/')
            vis_path = os.path.join(path, 'occ_vis_mlab/')
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)
            len_sequence = np.minimum(900, len(os.listdir(occ_path)))

            for i in range(len_sequence):
                fov_voxels = np.load(os.path.join(occ_path, 'occupancy_gt_with_semantic{}.npy'.format(i)))
                figure = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
                # pdb.set_trace()
                # fov_voxels = fov_voxels[fov_voxels[:,0]>99]
                # print(np.unique(fov_voxels[:, 3]))

                plt_plot_fov = mlab.points3d(
                    fov_voxels[:, 0],
                    fov_voxels[:, 1],
                    fov_voxels[:, 2],
                    fov_voxels[:, 3],
                    colormap="cool",
                    scale_factor=voxel_size,
                    mode="cube",
                    opacity=1.0,
                    vmax = 15,
                    vmin= 0
                )

                plt_plot_fov.glyph.scale_mode = "scale_by_vector"
                plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors_vod
                mlab.view(180, 60, 500)
                # mlab.yaw(45)



                # mlab.show()
                mlab.savefig(os.path.join(vis_path, str(i).zfill(9) + '.png'))
                mlab.clf()
                mlab.close()





if __name__ == '__main__':
    main()
