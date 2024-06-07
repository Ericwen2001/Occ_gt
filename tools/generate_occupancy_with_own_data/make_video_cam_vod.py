import os
import yaml
from glob import glob
from vod.common import get_frame_list
import moviepy.video.io.ImageSequenceClip


def main():
    fps=10
    root_dir = '/mnt/data/fangqiang/view_of_delft/lidar/training/image_2/'
    clip_dir =  "/home/xiangyu/SurroundOcc/tools/generate_occupancy_with_own_data/clips/"
    save_dir = '/mnt/data/fangqiang/vod_occ_format/cam_front_vis/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open('/home/xiangyu/SurroundOcc/tools/generate_occupancy_with_own_data/clips_info.yaml','r') as f:
        splits = yaml.safe_load(f.read())
    for split in splits:
        for clip in splits[split]:
            frames = get_frame_list(clip_dir + '/' + clip + '.txt')
            image_files = []
            for frame in frames:
                image_files.append(os.path.join(root_dir, frame+'.jpg'))
            save_path = os.path.join(save_dir,'{}.mp4'.format(clip))
            if (not os.path.exists(save_path)) and (len(image_files)>0):
                clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
                clip.write_videofile(save_path)



if __name__ == '__main__':
    main()
