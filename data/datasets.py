import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.nn.functional import interpolate
from torchvision.transforms.functional import center_crop, resize


import os
import numpy as np
import scipy.io as sio


# HyperSpectral Image dataset
class HSIDataSet(Dataset):
    def __init__(self, args):
        
        hsi = sio.loadmat(args.data_path)
        key = list(hsi.keys())
        self.hsi = torch.tensor(hsi[key[-1]].astype('float32'))
        
        # Resize the input video and center crop
        self.crop_list, self.resize_list = args.crop_list, args.resize_list  
        first_frame = self.img_transform(self.img_load(0))
        self.final_size = first_frame.size(-2) * first_frame.size(-1)
        
        # shuffle ablation
        self.shuffle = args.shuffle
        self.newid = torch.randperm(self.hsi.shape[2])
    
    def img_load(self, idx):
        if isinstance(self.hsi, list):
            img = read_image(self.hsi[:,:,idx])
        else:
            img = self.hsi[:,:,idx].unsqueeze(2).permute(-1,0,1)
        img = img.numpy()
        
        # normalization
        data_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = torch.tensor(data_norm)
        return img
    
    def img_transform(self, img):
        if self.crop_list != '-1': 
            crop_h, crop_w = [int(x) for x in self.crop_list.split('_')[:2]]
            if 'last' not in self.crop_list:
                img = pad(img, crop_h, crop_w)
        if self.resize_list != '-1':
            if '_' in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split('_')]
                img = interpolate(img, (resize_h, resize_w), 'bicubic')
            else:
                resize_hw = int(self.resize_list)
                img = resize(img, resize_hw,  'bicubic')
        if 'last' in self.crop_list:
            img = center_crop(img, (crop_h, crop_w))
        return img
    
    def __len__(self):
        return self.hsi.shape[2]
    
    def __getitem__(self, idx):
        idx1 = self.newid[idx]
        tensor_image = self.img_transform(self.img_load(idx1))
        
        if self.shuffle:
            norm_idx = float(idx) / self.hsi.shape[2]
            sample = {'img': tensor_image, 'idx': idx, 'norm_idx': norm_idx}
        else:
            norm_idx = float(idx1) / self.hsi.shape[2]
            sample = {'img': tensor_image, 'idx': idx1, 'norm_idx': norm_idx}
        
        return sample

def pad(x, H, W):
    h, w = x.size(1), x.size(2)
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )