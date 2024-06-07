import os, sys
import cv2, imageio
import numpy as np
import torch
from glob import glob
import open3d as o3d
from tqdm import tqdm
import open3d as o3d
import scipy.io
from pathos.multiprocessing import ProcessingPool as Pool
from matplotlib import pyplot as plt



def generate_radar_pc(cube_file, save_dir):

    mat_data = scipy.io.loadmat(cube_file)

    return 0


def main():

    root_dir = "/mnt/data/DataSet/K-Radar/"
    pool = Pool(8)
    source_dirs = sorted(os.listdir(root_dir))
    for source in source_dirs:
        cube_dir = os.path.join(root_dir, source, 'radar_zyx_cube')
        save_dir = os.path.join(root_dir, source, 'radar_pc')
        if os.path.exists(cube_dir):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cube_files = sorted(glob(cube_dir + '/'+ '*.mat'))
            zip_save_dir = [save_dir for i in range(len(cube_files))]
            with tqdm(total=len(cube_files)) as pbar:
                for _ in pool.imap(generate_radar_pc, cube_files, zip_save_dir):
                    pbar.update()
            



if __name__ == '__main__':
    main()
