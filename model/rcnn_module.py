import torch, time
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from utils import down2n
from fcos import DETHead

# receives pooled features and predicts classes + refined boxes
class RCNN_Module(nn.Module):
    def __init__(self,cfg):
        super(type(self), self).__init__()
        # params
        self.head_oproi = cfg.head_oproi
        self.im_size = cfg.im_size
        self.map_size = (down2n(cfg.im_size[0],cfg.conv_npool[-1]),down2n(cfg.im_size[1],cfg.conv_npool[-1]))
        self.scale_f = float(self.map_size[0]) / float(self.im_size[0])
        self.pool_size = cfg.roip_size
        # feat modulation layer
        self.conv_x = nn.Conv2d(cfg.head_nfeat, cfg.head_nfeat, 1)
        self.conv_y = nn.Conv2d(cfg.head_nfeat, cfg.head_nfeat, 1)
        # detection head
        self.rcnn_head = DETHead(cfg)
        
        # init
        rcnn_convs = nn.ModuleList([self.conv_x, self.conv_y])
        for m in rcnn_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    
    def forward(self, feats, boxes):
        # given query feature, candidate features, candidate boxes
        # return classification results and bbox refinements
        
        # feats = (xfa, zfa)
        # xfa = list of len=2 (pos/neg) where xfa[0]= Tensor[bnum, ndim, pool_size,pool_size]
        # zfa = list of len=batch_size where zfa[i]=[nms_num_bb, ndim, pool_size, pool_size]
        # boxes = list of len=batch_size where boxes[i]=[num_num_bb, 4]
        
        # pos_feats [bnum, cnum, pool_sz, pool_sz]
        # neg_feats [bnum, nnum, cnum, pool_sz, pool_sz]
        xfa_p, xfa_n, yfa = feats
        pf = xfa_p  # use spatially pooled feats
        nf = xfa_n
        # candidate feats [bnum, bbnum, cnum, psz, psz]
        cf = yfa
        # store shapes
        batch_size = cf.shape[0]
        bbnum_size = cf.shape[1]
        nfeat_size = cf.shape[2]
        negff_size = nf.shape[1] if nf is not None else 0
        
        # feature modulation
        pf = self.conv_x(pf)
        nf = self.conv_x(nf.flatten(0,1)).view(batch_size, negff_size, nfeat_size, self.pool_size, self.pool_size) if nf is not None else None
        cf = self.conv_y(cf.flatten(0,1)).view(batch_size, bbnum_size, nfeat_size, self.pool_size, self.pool_size)
        
            # == for positive feats
        # repeat pf feats
        pf_r = pf.unsqueeze(1).repeat_interleave(bbnum_size, dim=1) # [bnum, bbnum, cnum, psz, psz]
        # multiply between feats (correlation) or concat channel dim
        cc = pf_r * cf #torch.cat((pf_r, cf), dim=2)#
        # detection head
        cl_p, op, re = self.rcnn_head(cc.flatten(0,1))
        cl_p = cl_p.view(batch_size, bbnum_size, 1, 2)
        op = op.view(batch_size, bbnum_size, 1) if self.head_oproi else None
        re = re.view(batch_size, bbnum_size, 4)
        #re = torch.zeros_like(re)
        
            # == for negative feats
        if nf is not None:
            nf_r = nf.unsqueeze(1).repeat_interleave(bbnum_size, dim=1) # [bnum, bbnum, nnum, cnum, psz, psz]
            cf_r = cf.unsqueeze(2).repeat_interleave(negff_size, dim=2) # [bnum, bbnum, nnum, cnum, psz, psz]
            cn = nf_r * cf_r  # correlation
#             cn = torch.cat((nf_r,cf_r), dim=3)   # concatenation
            # detection head
            cl_n, _, _ = self.rcnn_head(cn.flatten(0,2), out_re=False)
            cl_n = cl_n.view(batch_size, bbnum_size, negff_size, 2)
        
        # integrated classification scores [bnum, bbnum, 1+nnum, 2]
        cl = torch.cat((cl_p, cl_n), dim=2) if nf is not None else cl_p
        
        # == modify input boxes accto re output
        # boxes = [bnum, bbnum, x0y0x1y1]
        boxes = torch.stack(boxes)
        #bb = boxes + re
        # change to [bnum, bbnum, x_cen/y_cen/width/height]
        boxes_w  = boxes[...,2] - boxes[...,0]
        boxes_h  = boxes[...,3] - boxes[...,1]
        boxes_xc = boxes[...,0] + boxes_w*0.5
        boxes_yc = boxes[...,1] + boxes_h*0.5
        # modify accto regression outputs
        boxes_xc_m = boxes_xc + boxes_w * re[...,0]
        boxes_yc_m = boxes_yc + boxes_h * re[...,1]
        boxes_w_m  = boxes_w  * torch.exp(re[...,2])
        boxes_h_m  = boxes_h  * torch.exp(re[...,3])
        # revert cooridates
        boxes_x0 = (boxes_xc_m - boxes_w_m*0.5).unsqueeze(-1)
        boxes_x1 = (boxes_xc_m + boxes_w_m*0.5).unsqueeze(-1)
        boxes_y0 = (boxes_yc_m - boxes_h_m*0.5).unsqueeze(-1)
        boxes_y1 = (boxes_yc_m + boxes_h_m*0.5).unsqueeze(-1)
        # concat
        bb = torch.cat([boxes_x0, boxes_y0, boxes_x1, boxes_y1], dim=-1)
        
        return cl, op, bb, boxes
    
    
    
    
    
    
    
    
    
    