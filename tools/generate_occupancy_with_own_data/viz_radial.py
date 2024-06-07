import numpy as np
import os
from radial_dataset import RADIal
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Rectangle
from torch.utils.data import Dataset, DataLoader, random_split,Subset
from pathos.multiprocessing import ProcessPool as Pool



def viz_data_sample(dataset, sample_id, img_dir, lpc_dir, rpc_dir, fft_dir, seg_dir):
    
    data = dataset.__getitem__(sample_id)
    image = data[0]
    boxes = data[5]
    laser_pc = data[3]
    radar_pc = data[2]
    radar_FFT = data[1]
    segmap = data[4]
    # viz rgb and bbox
    # fig, ax = plt.subplots()
    # ax.imshow(image)
    # for box in boxes:
    #     if(box[0]==-1):
    #         break # -1 means no object
    #     rect = Rectangle(box[:2]/2,(box[2]-box[0])/2,(box[3]-box[1])/2,linewidth=3, edgecolor='r', facecolor='none')
    #     ax.add_patch(rect)
    # plt.savefig(os.path.join(img_dir, str(sample_id).zfill(6)))
    # plt.close()
    # plt.clf
    # viz laser pc
    plt.plot(-laser_pc[:,1],laser_pc[:,0],'.')
    for box in boxes:
        if(box[0]==-1):
            break # -1 means no object
        plt.plot(box[4],box[5],'rs')
    plt.xlim(-20,20)
    plt.ylim(0,100)
    plt.grid()
    plt.savefig(os.path.join(lpc_dir, str(sample_id).zfill(6)))
    plt.close()
    plt.clf
    # viz radar pc
    plt.plot(-radar_pc[:,1],radar_pc[:,0],'.')
    for box in boxes:
        if(box[0]==-1):
            break # -1 means no object
        plt.plot(box[7],box[8],'ro')
    plt.xlim(-20,20)
    plt.ylim(0,100)
    plt.grid()
    plt.savefig(os.path.join(rpc_dir, str(sample_id).zfill(6)))
    plt.close()
    plt.clf
    # viz radar fft
    radar_FFT = radar_FFT[...,::2]+1j*radar_FFT[...,1::2]
    power_spectrum = np.sum(np.abs(radar_FFT),axis=2)
    plt.figure(figsize=(10,10))
    plt.imshow(np.log10(power_spectrum))
    for box in boxes:
        if(box[0]==-1):
            break # -1 means no object
        Range = box[9] * 512/103 # 512 range bins for 103m
        Azimuth = box[10]
        Doppler = box[11]
        plt.plot(Doppler,Range,'ro')
    plt.savefig(os.path.join(fft_dir, str(sample_id).zfill(6)))
    plt.close()
    plt.clf
    # viz seg map
    plt.imshow(segmap) # In polar coordinnates
    # Range resolution divided by 2
    # Azimuth angle crop to 448 and resolution divided by 2
    for box in boxes:
        if(box[0]==-1):
            break # -1 means no object
        Range = box[9]*512/103/2
        Azimuth = -box[10]/0.4
        plt.plot(segmap.shape[1]/2+Azimuth ,Range,'ro')
    plt.savefig(os.path.join(seg_dir, str(sample_id).zfill(6)))
    plt.close()
    plt.clf
    
def main():
    pool = Pool(16)
    root_dir = '/mnt/data/DataSet/RADIal/'
    save_dir = '/mnt/data/fangqiang/RADIal_pro/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_dir = os.path.join(save_dir, 'rgb')
    lpc_dir = os.path.join(save_dir, 'laser_pc')
    rpc_dir = os.path.join(save_dir, 'radar_pc')
    fft_dir = os.path.join(save_dir, 'radar_fft')
    seg_dir = os.path.join(save_dir, 'seg_map')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(lpc_dir):
        os.makedirs(lpc_dir)
    if not os.path.exists(rpc_dir):
        os.makedirs(rpc_dir)
    if not os.path.exists(fft_dir):
        os.makedirs(fft_dir)
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
    dataset = RADIal(root_dir = root_dir ,difficult=True)
    length = len(dataset)
    datasets = [dataset for i in range(length)]
    img_dirs = [img_dir for i in range(length)]
    lpc_dirs = [lpc_dir for i in range(length)]
    rpc_dirs = [rpc_dir for i in range(length)]
    fft_dirs = [fft_dir for i in range(length)]
    seg_dirs = [seg_dir for i in range(length)]
    with tqdm(total=length) as pbar:
        for _ in pool.imap(viz_data_sample, datasets, range(length), img_dirs, lpc_dirs, rpc_dirs, fft_dirs, seg_dirs):
            pbar.update()
    # for i in tqdm(range(length)):
    #     data = dataset.__getitem__(i)
    #     sample_id = int(dataset.sample_keys[i])
    #     viz_data_sample(data, sample_id)

if __name__ == "__main__":
    main()

