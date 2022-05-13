import torch
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision import transforms
from einops import rearrange

class TinyFFTImageNet(torch.utils.data.Dataset):
    def __init__(self, dataset, norm="L1", train=True):
        self.imgs = dataset

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img, label = self.imgs[idx]
        fft = torch.fft.rfft2(img, norm='forward')
        fft[:, :, 19:] *= 0
        fft[:, 19:-19, :] *= 0

        # start
        c, h, w = fft.shape
        x = torch.empty(c*4, h, w, dtype=torch.float)
        
        x[0:3, :, :] = fft.abs()
        x[3:6, :, :] = fft.angle()
        x[6:9, :, :] = fft.real
        x[9:, :, :] =  fft.imag

        #x = torch.fft.irfft2(fft, norm='forward')
       
        return x, label
        
