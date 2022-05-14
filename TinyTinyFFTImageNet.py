import torch
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision import transforms
from einops import rearrange

class TinyFFTImageNet(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms, norm="L1", train=True):
        self.imgs = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img, label = self.imgs[idx]
        img = self.transforms(img)
        return img, label
        
