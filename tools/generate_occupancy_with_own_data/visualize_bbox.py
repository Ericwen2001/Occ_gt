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
    def bbox_to_corners(x, y, z, l, w, h, theta):
    # Rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])

        # Half dimensions
        l, w, h = l/2, w/2, h/2

        # Original corner points
        corners = np.array([
            [-l, -w, -h],
            [-l, -w,  h],
            [-l,  w, -h],
            [-l,  w,  h],
            [ l, -w, -h],
            [ l, -w,  h],
            [ l,  w, -h],
            [ l,  w,  h]
        ]).T

        # Rotating and translating the points
        rotated_corners = R @ corners
        rotated_corners[0, :] += x
        rotated_corners[1, :] += y
        rotated_corners[2, :] += z

        return rotated_corners.T
        
    class Object3D():
        def __init__(self, xc, yc, zc, xl, yl, zl, rot_rad):
            self.xc, self.yc, self.zc, self.xl, self.yl, self.zl, self.rot_rad = xc, yc, zc, xl, yl, zl, rot_rad

            corners_x = np.array([xl, xl, xl, xl, -xl, -xl, -xl, -xl]) / 2 
            corners_y = np.array([yl, yl, -yl, -yl, yl, yl, -yl, -yl]) / 2 
            corners_z = np.array([zl, -zl, zl, -zl, zl, -zl, zl, -zl]) / 2 

            self.corners = np.row_stack((corners_x, corners_y, corners_z))

            rotation_matrix = np.array([
                [np.cos(rot_rad), -np.sin(rot_rad), 0.0],
                [np.sin(rot_rad), np.cos(rot_rad), 0.0],
                [0.0, 0.0, 1.0]])

            self.corners = rotation_matrix.dot(self.corners).T + np.array([[self.xc, self.yc, self.zc]])

    
    
    # process one clip once
    splits = ['train']
    for split in splits:
        # clips = sorted(os.listdir(os.path.join(root_path, split)), key=lambda x:int(x.split("_")[1]))
        for clip in ['38']:
            # if not clip == 'delft_13':
            #     continue 
            
            print('***Start to process {}***'.format(clip))
            path = os.path.join(root_path, split, clip)
            occ_path = os.path.join(path,'occupancy_gt/')
            vis_path = os.path.join(path, 'occ_vis/')
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)
            len_sequence = np.minimum(200, len(os.listdir(occ_path)))
            # vis = o3d.visualization.Visualizer()
            # vis.create_window(width=1280,height=720)
            print(len_sequence)
            i = 15
            fov_voxels = np.load(os.path.join(occ_path, 'occupancy_gt{}.npy'.format(i)))
            bbox = np.load("/mnt/data/DataSet/K-RadarOOC/train/"+ clip +"/bbox/bbox{}.npy".format(i))
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(fov_voxels)
            voxelGrid = voxel_down_sample(pcd, voxel_size=voxel_size)

            raw_pcd = o3d.io.read_point_cloud('/mnt/Kradar/K-Radar/'+ clip +'/os2-64/os2-64_00015.pcd')
            # pc = voxel_down_sample(pcd, voxel_size=voxel_size)

            # vis.add_geometry(voxelGrid)
            line_sets_bbox = []

            bboxes_o3d = []
            for b in bbox:
                
                x, y, z, l, w, h, theta = b
                l *= 1.6
                w *= 1.6
                h*=1.6
                bboxes_o3d.append(Object3D(x, y, z, l, w, h, theta+np.pi/4))
   
            for gt_obj in bboxes_o3d:
                lines = [[0, 1], [2, 3], # [0, 3], [1, 2],
                [4, 5], [6, 7],
                [0, 4], [1, 5], [2, 6], [3, 7],
                [0, 2], [1, 3], [4, 6], [5, 7]]
    
                colors = [[1, 0, 0] for i in range(len(lines))]  # red color lines
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(gt_obj.corners)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                line_sets_bbox.append(line_set)
                
                # vis.add_geometry([line_set])
            # ctr = vis.get_view_control()
            
            
            # vis.poll_events()
            # vis.update_renderer()
            # vis.run()
            o3d.visualization.draw_geometries([raw_pcd]+line_sets_bbox)
            # vis.capture_screen_image(os.path.join(images_outdir, "{}.png".format(file_name)))

            # vis.destroy_window()


if __name__ == '__main__':
    main()
