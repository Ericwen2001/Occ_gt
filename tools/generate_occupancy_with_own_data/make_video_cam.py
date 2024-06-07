import os
from glob import glob
import moviepy.video.io.ImageSequenceClip

fps=30
root_dir = '/mnt/Kradar/K-Radar/'
save_dir = '/mnt/data/DataSet/K-RadarOOC/cam_front_vis/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
source_dirs = sorted(os.listdir(root_dir))
# source_dirs = [root_dir]
test_dirs = ['21','38','39','42','45']
for source in test_dirs:
    image_dir = os.path.join(root_dir, source, 'cam-front')
    if os.path.exists(image_dir):
        image_files = sorted(glob(image_dir + '/'+ '*.png'))
        save_path = os.path.join(save_dir,'{}.mp4'.format(source))
        print(save_path)
        print(len(image_files))
        if (not os.path.exists(save_path)) and (len(image_files)>0):
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(save_path)


