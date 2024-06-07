import os, sys
import cv2, imageio
import numpy as np
import torch
from glob import glob
import open3d as o3d
from tqdm import tqdm
import open3d as o3d
from pathos.multiprocessing import ProcessingPool as Pool
from matplotlib import pyplot as plt

def viz_and_save(img_file, pcd_file, save_path):
    pcd = o3d.io.read_point_cloud(pcd_file)  
    # print('succeffuly read {}'.format(pcd_file))
    points = np.asarray(pcd.points)[::4]
    points = points[points[:,0]>0]
    ones = np.ones((points.shape[0], 1))
    homo_points = np.hstack((points, ones)).T
    transform_matrix =  np.array([[ 6.29238090e+02, -5.27036820e+02, -4.59938064e+00, 1.41321630e+03],
                         [ 3.72013478e+02,  9.42970300e+00, -5.59685543e+02, 1.07637845e+03],
                         [ 9.99925370e-01,  1.22165356e-02,     1.06612091e-04, 1.84000000e+00]])
    uvs = (transform_matrix @ homo_points).T
    threshold = np.percentile(uvs[:, 2], 0)
    uvs = uvs[uvs[:, 2] > threshold]
    uv = uvs[:, :2] / uvs[:, 2, np.newaxis]
    plt.rcParams["figure.figsize"] = (17,17)
    image = plt.imread(img_file)
    image_h, image_w = int(image.shape[0]), int(image.shape[1]/2)
    plt.imshow(image[:image_h, image_w:])
    valid_indices = (uv[:, 0] >= 0) & (uv[:, 0] < image_w) & \
                    (uv[:, 1] >= 0) & (uv[:, 1] < image_h)
    filtered_uv = uv[valid_indices]
    plt.scatter(filtered_uv[:,0], filtered_uv[:,1], c=uvs[valid_indices, 2], cmap='viridis', s= 0.1)
    plt.savefig(os.path.join(save_path, img_file.split('/')[-1]), dpi=200)
    plt.close()
    plt.clf()
    # print('succeffuly save{}'.format(os.path.join(save_path, img_file.split('/')[-1])))



def main():

    root_dir = "/mnt/data/DataSet/K-Radar/"
    save_dir = '/mnt/data/DataSet/K-RadarOOC/cam_pcd_vis/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pool = Pool(8)
    source_dirs = sorted(os.listdir(root_dir))
    for source in source_dirs:
        image_dir = os.path.join(root_dir, source, 'cam-front')
        pcd_dir = os.path.join(root_dir, source, 'os2-64')
        save_path = os.path.join(save_dir, source)
        if os.path.exists(pcd_dir):
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image_files = sorted(glob(image_dir + '/'+ '*.png'))[0::3]
            pcd_files = sorted(glob(pcd_dir + '/'+ '*.pcd'))[:len(image_files)]
            zip_save_path = [save_path for i in range(len(pcd_files))]
            with tqdm(total=len(pcd_files)) as pbar:
                for _ in pool.imap(viz_and_save, image_files, pcd_files, zip_save_path):
                    pbar.update()
            # tqdm(pool.imap(viz_and_save, image_files, pcd_files, zip_save_path))
            # for i in tqdm(range(len(pcd_files))):
            #     viz_and_save(image_files[i], pcd_files[i], save_path)







if __name__ == '__main__':
    main()
