import os
from glob import glob
import moviepy.video.io.ImageSequenceClip

fps=10
root_dir = '/mnt/data/fangqiang/vod_occ_format/val/'
save_dir = '//mnt/data/fangqiang/vod_occ_format/occ_vis_mlab/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
source_dirs = sorted(os.listdir(root_dir))
# source_dirs = [root_dir]
for source in source_dirs:
    image_dir = os.path.join(root_dir, source, 'occ_vis_mlab')
    if os.path.exists(image_dir):
        image_files = sorted(glob(image_dir + '/'+ '*.png'))
        save_path = os.path.join(save_dir,'{}.mp4'.format(source))
        if (len(image_files)>0):
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(save_path)


