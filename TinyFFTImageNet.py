import torch
import numpy as np
from tqdm import tqdm
import torchvision
from torchvision import transforms
from einops import rearrange

class TinyFFTImageNet(torch.utils.data.Dataset):
    def __init__(self, path, n_tokens, tok_dim, norm="L1", train=True):
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
        self.imgs = torchvision.datasets.CIFAR10("/raid/projects/weustis/data/cifar10", train=train,download=True, transform=pre_process)
        self.fft_shape = (64, 33)
        h, w = self.fft_shape
        funcs = {
            "L1":  self.l1_idx,
            "L2": self.l2_idx,
            "Linf": self.linf_idx
        }
        assert norm in funcs
        x = torch.zeros(self.fft_shape, dtype=torch.long)
        x2 = torch.zeros(self.fft_shape, dtype=torch.long)
        for r in range(h):
            for c in range(w):
                r_adj = -1*abs(r-112)+112
                x[r][c] = funcs[norm](r_adj, c)
        
        # todo pad w/ torch.nn.functional.pad so any tok_dim is allowed
        idx = x.int().flatten().argsort()
        self.tok_dim = tok_dim
        self.token_idx =  torch.arange(h*w)[idx] # .reshape((-1, tok_dim))
        
        self.max_n_tok = len(self.token_idx)//tok_dim
        self.means = torch.load('means.pt')[0]
        self.std = torch.load('std.pt')[0]
 
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img, label = self.imgs[idx]
        fft = torch.fft.rfft2(img, norm='forward')
        fft = fft.reshape(1,3,-1)
        fft = fft[:, :, self.token_idx] # reorder 
        fft = rearrange(fft, 'b c (s d) -> b s (c d)', d=self.tok_dim)[0]
        fft = fft[:self.n_tokens]
        s, e = fft.shape
        x = torch.empty(s, e*4, dtype=torch.float)
        x[:, :e] = fft.real
        x[:, e:2*e] = fft.imag
        x[:, 2*e:3*e] = fft.abs()
        x[:, 3*e:] = fft.angle()
        x = x - self.means[:s]
        x = x / self.std[:s]
        return x, label
        
    def l1_idx(self, r,c):
        cpn_r = r*(r+1)//2
        cpn_c = c*(c+1)//2
        return cpn_r + cpn_c + r*c + r
    def l2_idx(self, r,c):
        return (c**2 + r**2)*self.fft_shape[0]*self.fft_shape[1] + r*c + r
    def linf_idx(self, r,c):
        return max(r, c)*self.fft_shape[0]*self.fft_shape[1] + r*self.fft_shape[0] + c
