import torch
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision import transforms
from einops import rearrange

class TinyFFTImageNet(torch.utils.data.Dataset):
    def __init__(self, path, n_tokens, tok_dim, norm="Linf", train=True):
        self.n_tokens = n_tokens
        pre_process = transforms.Compose([
             transforms.Resize(74),
             transforms.CenterCrop(64),
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

        self.max_n_tok  = 64//4 * 33//3

        self.patch_size = (4,3)



    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img, label = self.imgs[idx]
        fft = torch.fft.rfft2(img, norm='forward')
        fft = rearrange(fft, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1 = 4, p2 = 3)
       # c, h, w = fft.shape
        s, e = fft.shape
        x = torch.empty(s, e*4, dtype=torch.float)
        
        x[:, ::4] = fft.real
        x[:, 1::4] = fft.imag
        x[:, 2::4] = fft.abs()
        x[:, 3::4] = fft.angle()
        # x = x - self.means[:s]
        # x = x / self.std[:s]
        return x, label
        
    def l1_idx(self, r,c):
        cpn_r = r*(r+1)//2
        cpn_c = c*(c+1)//2
        return cpn_r + cpn_c + r*c + r
    def l2_idx(self, r,c):
        return (c**2 + r**2)*self.fft_shape[0]*self.fft_shape[1] + r*c + r
    def linf_idx(self, r,c):
        return max(r, c)*self.fft_shape[0]*self.fft_shape[1] + r*self.fft_shape[0] + c
