import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import nms
from time import time

from th_utils import generate_reg_coords

# box prediction module given response maps
class BoxModule(nn.Module):
    def __init__(self, cfg):
        super(type(self), self).__init__()
        self.im_size = cfg.im_size
        # nms parameters
        self.bb_thres = cfg.nms_param[0]
        self.bb_nums  = cfg.nms_param[1]
        # default anchor box center coordinates
        self.anc  = torch.Tensor(generate_reg_coords(cfg)).unsqueeze(0).flatten(1,-2).cuda()

    
    def forward(self,cl,re, nms_param=None):
        # define nms parameters
        if nms_param is not None:
            bb_thr = nms_param[0]
            bb_num = nms_param[1]
        else:
            bb_thr = self.bb_thres
            bb_num = self.bb_nums
        
        # softmax class -> obtain scoremap
        ff = torch.exp(cl[...,0]) / (torch.exp(cl[...,0])+torch.exp(cl[...,1])) # [bnum, map_h, map_w]
        batch_size = ff.shape[0]
        # flatten scoremaps and regvals
        ff_f = ff.flatten(1) # [bnum,N]
        re_f = re.flatten(1,-2) #[bnum,N,ltrb]
        
        # translate regressed vals to bbox coordinates [bnum, N, x0y0x1y1]
        bb_f = self.anc.repeat_interleave(batch_size,dim=0).clone() # anchor coordinates to xyxy [bnum,N,xyxy]
        bb_f[...,0] -= re_f[...,0] # x_min = x_anc - left
        bb_f[...,1] -= re_f[...,1] # y_min = y_anc - top
        bb_f[...,2] += re_f[...,2] # x_max = x_anc + right
        bb_f[...,3] += re_f[...,3] # y_max = y_anc + down
        
        # cutoff boundary values
        xmin,ymin,xmax,ymax = bb_f[...,0],bb_f[...,1],bb_f[...,2],bb_f[...,3]
        xmin[xmin<0] = 0
        ymin[ymin<0] = 0
        xmax[xmax>self.im_size[1]-1] = self.im_size[1]-1
        ymax[ymax>self.im_size[0]-1] = self.im_size[0]-1
        
        # per-batch nms
        out_bb, out_ff = [], []
        for i in range(batch_size):
            ffi = ff_f[i]
            bbi = bb_f[i]
            b_idx = nms(bbi, ffi, bb_thr)
            # if numof boxes to choose is larger than obtained numof boxes
            b_sel = torch.LongTensor(range(bb_num)).cuda()
            b_sel[b_sel>len(b_idx)-1] = len(b_idx)-1
            # choose and store boxes
            b_box = bbi[b_idx[b_sel]]
            out_bb.append(b_box)
            out_ff.append(ffi[b_idx[b_sel]])

        # output : list of boxes where len(list)=batch_size, list[i]=[num_box,xyxy]
        return out_bb, out_ff
            
            
            
