import torch
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision import transforms
from einops import rearrange

class TinyFFTImageNet(torch.utils.data.Dataset):
    def __init__(self, path, n_tokens, tok_dim, patch_size, norm="L1", train=True):
        self.n_tokens = n_tokens
        pre_process = transforms.Compose([
             transforms.Resize(74),
             transforms.RandomCrop(64),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(5),
             transforms.RandAugment(),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        if train==False:
            pre_process = transforms.Compose([
             transforms.Resize(74),
             transforms.CenterCrop(64),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.imgs = torchvision.datasets.ImageFolder(path, transform=pre_process)
        self.fft_shape = (64, 33)
        h, w = self.fft_shape
        self.patch_size = patch_size
        self.max_n_tok  = self.fft_shape[0]//patch_size[0] * self.fft_shape[1]//patch_size[1]

        



    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img, label = self.imgs[idx]
       # fft = torch.fft.rfft2(img, norm='forward')
     #   fft = rearrange(fft, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1 = 4, p2 = 3)
      #  c, h, w = fft.shape
       # c, e = fft.shape
    #    x = torch.empty(c*2, h, w, dtype=torch.float)
        
     #   x[0:3, :, :] = fft.abs()#fft.real
     #   x[3:6, :, :] = fft.angle()#fft.imag
       # x[6:9, :, :] = 
       # x[9:, :, :] = 

        return x, label
        
    def l1_idx(self, r,c):
        cpn_r = r*(r+1)//2
        cpn_c = c*(c+1)//2
        return cpn_r + cpn_c + r*c + r
    def l2_idx(self, r,c):
        return (c**2 + r**2)*self.fft_shape[0]*self.fft_shape[1] + r*c + r
    def linf_idx(self, r,c):
        return max(r, c)*self.fft_shape[0]*self.fft_shape[1] + r*self.fft_shape[0] + c
