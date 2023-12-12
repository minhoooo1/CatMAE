from torch.utils.data import Dataset
from glob import glob
import random
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from skimage import io
import logging
from decord import VideoReader
from decord import cpu, gpu
import math
from torchvision.utils import flow_to_image
from torchvision.transforms import functional as F
from torchvision.transforms import RandomResizedCrop



class TripRandomResizedCrop:
    def __init__(self, hflip_p=0.5, size=(224, 224), scale=(0.5, 1.0), ratio=(3./4., 4./3.), interpolation=F.InterpolationMode.BICUBIC):
        self.hflip_p = hflip_p
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.resized_crop = RandomResizedCrop(size, scale, ratio, interpolation)
        
    def __call__(self, np_RGB_img_1, np_RGB_img_2, np_RGB_img_3):
        # Convert numpy images to PIL Images
        pil_RGB_img_1 = F.to_pil_image(np_RGB_img_1)
        pil_RGB_img_2 = F.to_pil_image(np_RGB_img_2)
        pil_RGB_img_3 = F.to_pil_image(np_RGB_img_3)
        
        # Apply the crop on both images
        cropped_img_1 = self.resized_crop(pil_RGB_img_1)
        cropped_img_2 = self.resized_crop(pil_RGB_img_2)
        cropped_img_3 = self.resized_crop(pil_RGB_img_3)
        
        if random.random() < self.hflip_p:
            cropped_img_1 = F.hflip(cropped_img_1)
            cropped_img_2 = F.hflip(cropped_img_2)
            cropped_img_3 = F.hflip(cropped_img_3)
            
        return cropped_img_1, cropped_img_2, cropped_img_3
    
    
class CustomKinetics400Dataset(Dataset):
    def __init__(self, video_dir, transform_triple, transform_totensor, frame_interval=[[4,48],[4,48]], repeated_sampling=2, use_vire=False):
        s = time.time()
        print("start to find videos")
        self.videos_path = glob(os.path.join(video_dir, "*.mp4"))
        print(f"{time.time()-s} alread find videos")
        
        self.transform_triple = transform_triple
        self.transform_totensor = transform_totensor
        
        self.frame_interval = frame_interval
        self.repeated_sampling = repeated_sampling
        self.use_vire = use_vire
        
    def __len__(self):
        return self.repeated_sampling*len(self.videos_path)


    def _get_frames_index(self, total_frames):
        least_frames_num = self.frame_interval[0][1] + self.frame_interval[1][1] + 1
        if total_frames >= (least_frames_num):
            frame1_idx = random.randint(0, total_frames - least_frames_num)
            
            interval_2 = random.randint(*self.frame_interval[0])
            frame2_idx = frame1_idx + interval_2
            
            interval_3 = random.randint(*self.frame_interval[1])
            frame3_idx = frame2_idx + interval_3
        elif total_frames >= 3:
            indices = random.sample(range(total_frames), 3)
            indices.sort()
            frame1_idx, frame2_idx, frame3_idx = indices

        else:
            assert False, f"frames index error"
        
        return frame1_idx, frame2_idx, frame3_idx
    
    
    def __getitem__(self, idx):
        reverse = idx % 2
        
        idx = idx//self.repeated_sampling
        video_path = self.videos_path[idx]
        
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        frame1_idx, frame2_idx, frame3_idx = self._get_frames_index(total_frames)
        if self.use_vire and reverse:
            frame1_idx, frame3_idx = frame3_idx, frame1_idx

        frame1, frame2, frame3 = vr[frame1_idx].asnumpy(), vr[frame2_idx].asnumpy(), vr[frame3_idx].asnumpy()
        # print(frame1_idx, frame2_idx, frame3_idx)
        frame1, frame2, frame3 = self.transform_triple(frame1, frame2, frame3)    # array resized RGB 
        frame1, frame2, frame3 = self.transform_totensor(frame1), self.transform_totensor(frame2), self.transform_totensor(frame3)
        return frame1, frame2, frame3


if __name__ == "__main__":
    transform_triple = TripRandomResizedCrop()
    transform_totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
    video_dir = "/home/cv/data/Kinetics-400/videos_train/"
    dataset = CustomKinetics400Dataset(video_dir ,transform_triple, transform_totensor)
    f1, f2, f3= dataset[0]
    # print(f1.shape, f2.shape, f, i)
    print(f1.shape, f2.shape, f3.shape)
    
    # print(f1.min(), f1.max(), f2.min(), f2.max(), )
    print(f1.min(), f1.max(), f2.min(), f2.max(), f3.min(), f3.max())
    
    f1 =  (f1 - f1.min()) / (f1.max() - f1.min())
    f2 =  (f2 - f2.min()) / (f2.max() - f2.min())
    f3 =  (f3 - f3.min()) / (f3.max() - f3.min())
    
    
    plt.subplot(1,3,1)
    plt.imshow(f1.permute(1,2,0).numpy())
    plt.subplot(1,3,2)
    plt.imshow(f2.permute(1,2,0).numpy())
    plt.subplot(1,3,3)
    plt.imshow(f3.permute(1,2,0).numpy())
    plt.savefig("33333.jpg")
