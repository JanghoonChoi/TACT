import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import down2n

# torch implementation of np.random.choice
def th_choice(a, p=None):
    """ torch implementation of np.random.choice(), x1.1~1.5 slower than original function """
    # preliminaries
    a_l = len(a)
    if p is None:
        idx = torch.randperm(a_l)
        return a[idx[0]]
        
    elif torch.sum(p) < 1.:
        print torch.sum(p),' p.sum() not 1'
    
    # accumulative prob
    pa = torch.cumsum(p,0)
    
    # random (0,1)
    trnd = torch.rand(1)[0]
    
    # find
    idx = (torch.argmax((pa < trnd).type(torch.FloatTensor))+1) % a_l
    return a[idx]


def th_choice_mul(a, n):
    # choose n random instances from a
    # assume p=uniform, with replacement
    a_l = len(a)
    idxs = torch.randint(low=0, high=a_l, size=(n,))
    
    if isinstance(a, list):
        return [a[i] for i in idxs]
    elif n==1:
        return [a[idxs]]
    else:
        return a[idxs]
    

def th_choice_seq(a, n):
    # choose n sequential instances from a
    # assume p=uniform, with replacement
    a_l = len(a)
    if n <= a_l:
        idx = torch.randint(low=0, high=a_l-n+1, size=())
        idxs = torch.LongTensor(range(idx, idx+n))
    else:
        idxs = torch.LongTensor(range(a_l)+[a_l-1]*(n-a_l))
    
    if isinstance(a, list):
        return [a[i] for i in idxs]
    elif n==1:
        return [a[idxs]]
    else:
        return a[idxs]


def th_rand(n=1):
    """ proxy to torch.rand(n)[0] """
    if n == 1:
        return float(torch.rand(n)[0])
    else:
        return torch.rand(n).numpy()

    
def th_rand_rng(low, high, n=1):
    """ pull uniform random sample(s) from [a,b) """
    if n == 1:
        return (high-low)*float(torch.rand(n)[0])+low
    else:
        return (high-low)*torch.rand(n)+low


def th_rand_sym(r, n=1):
    """ pull random sample(s) from [1/r,r), keeping probability mean to 1.0 """
    def unit_rnd(r):
        ud_rf = 1 if th_rand() < 1./(r+1.) else 0
        rnd = th_rand_rng(1.,r) if ud_rf else th_rand_rng(1./r,1)
        return rnd
    
    if n == 1:
        return unit_rnd(r)
    else:
        return torch.Tensor([unit_rnd(r) for i in range(n)])
    
    
def th_randint(low, high=None, size=1):
    """ proxy to torch.randint(low,high,(size,)) """
    if high is None:    ilow = 0;    ihigh = low
    else:    ilow = low;    ihigh = high
        
    if size == 1:
        return torch.randint(low=ilow, high=ihigh, size=(size,)).numpy()[0]
    else:
        return torch.randint(low=ilow, high=ihigh, size=(size,)).numpy()

    
# generate center-anchor cooridnates for a given img_size and pooling size
def generate_reg_coords(cfg):
    map_size = (down2n(cfg.im_size[0],cfg.conv_npool[-1]),down2n(cfg.im_size[1],cfg.conv_npool[-1]))
    
    batch_gtr = np.zeros([map_size[0], map_size[1], 4]) #[map_h, map_w, ltrb]
    grid_r = np.tile(np.arange(0.5, 0.5+map_size[0], 1.).reshape([-1,1]),(1,map_size[1]))
    grid_c = np.tile(np.arange(0.5, 0.5+map_size[1], 1.).reshape([1,-1]),(map_size[0],1))
    map_scale = float(cfg.im_size[0])/float(map_size[0])

    batch_gtr[:,:,0] = grid_c # left
    batch_gtr[:,:,1] = grid_r # top
    batch_gtr[:,:,2] = grid_c # right
    batch_gtr[:,:,3] = grid_r # bottom
    batch_gtr *= map_scale # rescale map by size

    return batch_gtr
    
    
