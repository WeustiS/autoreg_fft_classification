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
       
        return img, label
        
