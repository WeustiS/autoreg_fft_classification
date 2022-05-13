import torch
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision import transforms
from einops import rearrange

class TinyFFTImageNet(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms=None, norm="L1", train=True):
        self.imgs = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img, label = self.imgs[idx]
        if transforms is not None:
            img = self.transforms(img)
        fft = torch.fft.rfft2(img, norm='forward')
        c, h, w = fft.shape
        x = torch.empty(c*4, h, w, dtype=torch.float)
        
        x[0:3, :, :] = fft.abs()
        x[3:6, :, :] = fft.angle()
        x[6:9, :, :] = fft.real
        x[9:, :, :] =  fft.imag

        return x, label
        
