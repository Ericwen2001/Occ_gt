import os
import numpy as np
import mayavi.mlab as mlab

def box_center_to_corner(box):
    translation = box[0:3]
    l, w, h = box[3], box[4], box[5]
    rotation = box[6]
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [0,0,0,0, h, h, h, h]])
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])
    eight_points = np.tile(translation, (8, 1))
    corner_box = np.dot(rotation_matrix, bounding_box) + eight_points.transpose()
    return corner_box.transpose()

def main():
    colors = np.array(
        [   [0, 0, 0, 255], # unoccupied  white
            [150, 255, 150, 255], #background
            [255, 1, 7, 255],  # foreground blue
        ]
    ).astype(np.uint8)

    voxel_size = 0.95
    pc_range = [-50, -50, -5, 50, 50, 5]
    root_path = "/mnt/data/DataSet/occ_work_dir/RadarOcc_self/viz"
    save_img = True
    
    splits = ['3']  # Assuming we only have split '3'
    for split in splits:
        clips = sorted(
            [clip for clip in os.listdir(os.path.join(root_path, split)) if clip.isdigit()],
            key=lambda x: int(x)
        )
        for clip in clips:
            print('clip', clip)
            print('***Start to process {}***'.format(clip))
            path = os.path.join(root_path, split, clip)
            vis_path = '/mnt/data/DataSet/viz/RadarOOC_pred/{}'.format(split)
            os.makedirs(vis_path, exist_ok=True)
  
            len_sequence = np.minimum(300, len(os.listdir(path)))

            mlab.figure(size=(1280, 720), bgcolor=(1, 1, 1))

            plt_plot_fov = None

            pred_voxels = np.load(os.path.join(path, f'pred_c.npy'))
            
            if pred_voxels.size == 0:
                continue

            # Filter class indices
            pred_voxels[:, 3] = np.where(pred_voxels[:, 3] > 2, 2, pred_voxels[:, 3])
            
            if plt_plot_fov is None:
                plt_plot_fov = mlab.points3d(
                    -pred_voxels[:, 0],
                    pred_voxels[:, 1],
                    pred_voxels[:, 2],
                    pred_voxels[:, 3],
                    colormap="cool",
                    scale_factor=voxel_size,
                    mode="cube",
                    opacity=1.0,
                    vmax=2,
                    vmin=0
                )
                plt_plot_fov.glyph.scale_mode = "scale_by_vector"
                plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
                mlab.view(azimuth=180, elevation=155,  distance=100, focalpoint=(0,65,50))
            else:
                plt_plot_fov.mlab_source.reset(
                    x=pred_voxels[:, 0],
                    y=pred_voxels[:, 1],
                    z=pred_voxels[:, 2],
                    scalars=pred_voxels[:, 3]
                )

            if save_img:
                print(f'Saving frame {clip} to {vis_path}')
                mlab.savefig(os.path.join(vis_path, str(clip).zfill(9) + '.png'))

            # mlab.process_ui_events()
            
            mlab.close()  # Close the figure to prevent freezing for the next clip

if __name__ == '__main__':
    main()
