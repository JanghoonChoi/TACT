import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from torchvision.ops import roi_align

from utils import down2n
from box_utils import jaccard
from boxes import BoxModule

# context feature embedding module (input: rois+pooled feats from rpn -> output: rois+context embedded features -> to rcnn module)
class ContextModule(nn.Module):
    def __init__(self,cfg):
        super(type(self), self).__init__()
        # params
        self.im_size = cfg.im_size
        self.map_size = (down2n(cfg.im_size[0],cfg.conv_npool[-1]),down2n(cfg.im_size[1],cfg.conv_npool[-1]))
        self.scale_f = float(self.map_size[0]) / float(self.im_size[0])        
        self.pool_size  = cfg.roip_size
        self.head_nfeat = cfg.head_nfeat
        self.head_ctxff = cfg.head_ctxff
        self.num_ctxff  = cfg.nft_param[1]
        self.ctx_param  = (0.5,self.num_ctxff) #4
        # box module
        self.boxes = BoxModule(cfg)
        
        # variables w.r.t. different fusion schemes
        if self.head_ctxff[1]==0:
            fdim,reduce = (self.head_nfeat+2)*3-2, 1
            # simple concat
            self.simple = nn.Sequential(*[nn.Conv2d(fdim, self.head_nfeat, 3,1,1), nn.ReLU(), 
                                          nn.Conv2d(self.head_nfeat, self.head_nfeat, 3,1,1), nn.ReLU(),
                                          nn.Conv2d(self.head_nfeat, self.head_nfeat, 1,1,0)])
            
        elif self.head_ctxff[1]==1:
            fdim,reduce = (self.head_nfeat+2)*2, 1
            # simple addition
            self.simple = nn.Sequential(*[nn.Conv2d(fdim, self.head_nfeat, 3,1,1), nn.ReLU(),
                                          nn.Conv2d(self.head_nfeat, self.head_nfeat, 3,1,1), nn.ReLU(),
                                          nn.Conv2d(self.head_nfeat, self.head_nfeat, 1,1,0)])
                                        
        elif self.head_ctxff[1]==2:
            # attention (cbam) based
            fdim,reduce = (self.head_nfeat+2)*2, 1
            # channel attention branch
            self.avg_pool, self.max_pool = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
            self.conv1, self.conv2 = nn.Conv2d(fdim, self.head_nfeat//reduce, 1), nn.Conv2d(self.head_nfeat//reduce, self.head_nfeat, 1)
            self.relu1 = nn.ReLU()
            # spatial attention branch
            self.conv3 = nn.Conv2d(2, 1, 5, 1, 2)
            self.sigmoid = nn.Sigmoid()
            
        elif self.head_ctxff[1]==3:
            # film based
            fdim,reduce = (self.head_nfeat+2)*2, 1
            # common conv+relu
            self.conv1 = nn.Sequential(*[nn.Conv2d(fdim, self.head_nfeat//reduce, 3,1,1), nn.ReLU()])
            # channel multiplier gamma
            self.mult_g = nn.Parameter(torch.ones(1, self.head_nfeat, self.pool_size, self.pool_size))
            self.conv_g = nn.Sequential(*[nn.Conv2d(self.head_nfeat//reduce, self.head_nfeat//reduce, 3,1,1), nn.ReLU(),
                                          nn.Conv2d(self.head_nfeat//reduce, self.head_nfeat, 1,1,0)])
            # channel bias beta
            self.mult_b = nn.Parameter(torch.zeros(1, self.head_nfeat, self.pool_size, self.pool_size))
            self.conv_b = nn.Sequential(*[nn.Conv2d(self.head_nfeat//reduce, self.head_nfeat//reduce, 3,1,1), nn.ReLU(),
                                          nn.Conv2d(self.head_nfeat//reduce, self.head_nfeat, 1,1,0)])
            
        else:
            print 'unknown fusion scheme...'
            
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0., 1e-3)
                nn.init.constant_(m.bias, 0.)
        
            
        
    def forward(self, cl,re, cf,yfa,ybb):
        # cl,re : for obtaining context boxes
        # cf    : full context feature map to sample features from
        # yfa   : input features to be embedded with context; [num_batch, num_boxes, num_ch, pool_sz, pool_sz]
        # ybb   : bounding box coordinates for input yfa feats; len(list)=num_batch, ybb[i]=[num_boxes,4]
        num_batch = yfa.shape[0]
        num_boxes = yfa.shape[1]
        # obtain candidate context box coordinates and pool feats cfa_all=[num_batch, num_ctx, num_ch, pool_sz, pool_sz]
        pred_ctxbb, pred_ctxsc = self.boxes(cl,re, self.ctx_param)
        ff = torch.cat((cf,cl.permute(0,3,1,2)),dim=1) # concat feats and cls logits
        cfa_all = roi_align(ff, pred_ctxbb, (self.pool_size,self.pool_size), self.scale_f)
        cfa_all = cfa_all.view(num_batch, self.ctx_param[1], self.head_nfeat+2, self.pool_size, self.pool_size)
        # max/mean pooling along channel dimension
        cfa_max,_ = cfa_all.max(dim=1)
        cfa_avg   = cfa_all.mean(dim=1)
        cfa = torch.cat((cfa_max,cfa_avg), dim=1)    # [num_batch, num_ch*2, pool_sz, pool_sz]
        
        # embed context into input feat yfa
        if self.head_ctxff[1]==0:
            # === simple concat
            cfa = cfa.unsqueeze(1).repeat_interleave(num_boxes,dim=1)# [num_batch, num_boxes, num_ch*2, pool_sz, pool_sz]
            cfa = torch.cat((yfa,cfa), dim=2) # channel-wise concat # [num_batch, num_boxes, num_ch*3, pool_sz, pool_sz]
            cfa = cfa.flatten(0,1) # batch-nbox dim flatten
            yfa = self.simple(cfa)
            yfa = yfa.view(num_batch, num_boxes, self.head_nfeat, self.pool_size, self.pool_size)
            
        elif self.head_ctxff[1]==1:
            # === simple addition
            cfa = self.simple(cfa) # [num_batch, self.head_nfeat, self.pool_size, self.pool_size]
            cfa = cfa.unsqueeze(1).repeat_interleave(num_boxes,dim=1)
            yfa += cfa
            
        elif self.head_ctxff[1]==2:
            # === channel and spatial attention (cbam) based
            # channel attention
            avg_out = self.conv2( self.relu1( self.conv1( self.avg_pool(cfa) ) ) )
            max_out = self.conv2( self.relu1( self.conv1( self.max_pool(cfa) ) ) )
            ca_out = self.sigmoid( avg_out + max_out )
            ca_out = ca_out.unsqueeze(1).repeat_interleave(num_boxes,dim=1)
            yfa *= ca_out
            # spatial attention
            avg_out   = torch.mean(cfa, dim=1, keepdim=True)
            max_out,_ =  torch.max(cfa, dim=1, keepdim=True)
            sp_out = torch.cat((avg_out,max_out), dim=1)
            sp_out = self.sigmoid( self.conv3(sp_out) )
            sp_out = sp_out.unsqueeze(1).repeat_interleave(num_boxes,dim=1)
            yfa *= sp_out
            
        elif self.head_ctxff[1]==3:
            # === film based affine transform
            # common branch
            fconv = self.conv1(cfa)
            # get channel multipler (mult_g*conv_g)
            fm_out = self.mult_g*self.conv_g(fconv)
            fm_out = fm_out.unsqueeze(1).repeat_interleave(num_boxes,dim=1)
            # get channel bias (mult_b*conv_b)
            fb_out = self.mult_b*self.conv_b(fconv)
            fb_out = fb_out.unsqueeze(1).repeat_interleave(num_boxes,dim=1)
            # apply channel wise linear transform ( (1-gamma)*feat+beta )
            yfa = (1+fm_out)*yfa + fb_out
            
        else:
            print 'unknown fusion scheme...'
        
        return yfa
        

        