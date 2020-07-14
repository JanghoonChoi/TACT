import torch
import torch.nn as nn
import torch.nn.functional as F

from non_local import NONLocalBlock2D


# fcos detection head module
class FCOSHead(nn.Module):
    def __init__(self, cfg):
        super(type(self), self).__init__()
        
        # define individual modules
        fdim = cfg.head_nfeat
        fmul = 3 if cfg.head_dconv else 1
        
        self.conv0_rdim = nn.Conv2d(fdim*fmul,fdim, 1,1,0)
        
        if cfg.head_nlocl:
            self.nl_feature = NONLocalBlock2D(in_channels=fdim)
        
        conv1_unit = [nn.Conv2d(fdim,fdim, 1,1,0), nn.GroupNorm(16,fdim), nn.ReLU()]
        conv2_unit = [nn.Conv2d(fdim,fdim, 1,1,0), nn.GroupNorm(16,fdim), nn.ReLU()]
        for i in range(cfg.head_nconv-1):
            conv1_unit.extend([nn.Conv2d(fdim,fdim, 1,1,0), nn.GroupNorm(16,fdim), nn.ReLU()]) #nn.Conv2d(fdim,fdim, 3,1,1)
            conv2_unit.extend([nn.Conv2d(fdim,fdim, 1,1,0), nn.GroupNorm(16,fdim), nn.ReLU()])
        
        self.conv1 = nn.Sequential(*conv1_unit)
        self.conv2 = nn.Sequential(*conv2_unit)
        
        self.conv_cls = nn.Sequential(nn.Conv2d(fdim,2, 3,1,1))
        self.conv_reg = nn.Sequential(nn.Conv2d(fdim,4, 3,1,1))
        
        # define sequential modules
        self.cls = nn.Sequential(self.conv1, self.conv_cls)
        self.reg = nn.Sequential(self.conv2, self.conv_reg)
        self.mul = nn.Parameter(torch.rand(1))
        
        # init
        head_module_list = nn.ModuleList([self.conv0_rdim, self.cls, self.reg])
        for m in head_module_list.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0., 1e-3)
                nn.init.constant_(m.bias, 0.)        
                
    
    def forward(self, x):
        # reduce dim
        x = self.conv0_rdim(x)
        # nonlocal
        if hasattr(self, 'nl_feature'):
            x = self.nl_feature(x)
        # for all branches
        cl = self.cls(x)
        re = torch.exp(self.mul*self.reg(x))
            
        return cl, re, x

    

# standard detection head module (cls, olp, reg)
class DETHead(nn.Module):
    def __init__(self, cfg):
        super(type(self), self).__init__()
        
        # define individual modules
        self.head_oproi = cfg.head_oproi
        fdim = cfg.head_nfeat
        conv1_unit = [nn.Conv2d(fdim,fdim, 1,1,0), nn.GroupNorm(16,fdim), nn.ReLU()]
        conv2_unit = [nn.Conv2d(fdim,fdim, 1,1,0), nn.GroupNorm(16,fdim), nn.ReLU()]
        for i in range(cfg.head_nconv-1):
            conv1_unit.extend([nn.Conv2d(fdim,fdim, 1,1,0), nn.GroupNorm(16,fdim), nn.ReLU()]) #nn.Conv2d(fdim,fdim, 3,1,1)
            conv2_unit.extend([nn.Conv2d(fdim,fdim, 1,1,0), nn.GroupNorm(16,fdim), nn.ReLU()])
        
        self.conv1 = nn.Sequential(*conv1_unit)
        self.conv2 = nn.Sequential(*conv2_unit)
        
        self.conv_cls = nn.Sequential(nn.Conv2d(fdim,2, cfg.roip_size,1,0))
        self.conv_reg = nn.Sequential(nn.Conv2d(fdim,4, cfg.roip_size,1,0))
        if self.head_oproi:
            self.conv_olp = nn.Sequential(nn.Conv2d(fdim,1, cfg.roip_size,1,0))
        
        # define sequential modules
        self.cls = nn.Sequential(self.conv1, self.conv_cls)
        self.reg = nn.Sequential(self.conv2, self.conv_reg)
        if self.head_oproi:
            self.olp = nn.Sequential(self.conv2, self.conv_olp)
        
        # init
        head_module_list = nn.ModuleList([self.cls, self.olp, self.reg]) if self.head_oproi else nn.ModuleList([self.cls, self.reg])
        for m in head_module_list.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0., 1e-3)
                nn.init.constant_(m.bias, 0.)
        
    
    def forward(self, x, out_re=True):
        # for all 3 branches
        cl = self.cls(x)
        op = self.olp(x) if (out_re and self.head_oproi) else None
        re = self.reg(x) if out_re else None
        
        return cl, op, re

    
    