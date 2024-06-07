import os
import glob
import numpy as np
splits = ['train']
data_path = '/mnt/data/DataSet/K-RadarOOC/'
for split in splits:
        clips = sorted(os.listdir(os.path.join('/mnt/data/DataSet/K-RadarOOC', split)), key=lambda x:int(x))
        clips = [1,2,3,4,6,8,9,10,11,12,13,14,15,19]
        for clip in clips:
            path = os.path.join(data_path, split, str(clip))
            gt_names = os.listdir( os.path.join(path, 'semantic_occupancy_gt_fov/'))
            for gt_name in gt_names:
                gt = np.load(os.path.join(path, 'semantic_occupancy_gt_fov/',gt_name))
                gt[gt[:, -1] >= 3, -1] = 2
