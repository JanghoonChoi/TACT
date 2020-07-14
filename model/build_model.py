import torch, time
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from box_utils import jaccard
import resnet as resnet
from rpn_module import RPN_Module
from rcnn_module import RCNN_Module

class Track_Model(nn.Module):
    def __init__(self,cfg):
        super(type(self), self).__init__()
        # dims and flags
        self.head_nfeat = cfg.head_nfeat
        self.head_negff = cfg.head_negff
        self.head_oproi = cfg.head_oproi
        self.head_ctxff = cfg.head_ctxff
        self.roip_size = cfg.roip_size
        self.nft_param = cfg.nft_param
        # backbone convnet
        self.backbone = getattr(resnet, cfg.name_bbnet)(cfg=cfg)
        # channel dim for backbone output featmap
        bb_ch = self.backbone(torch.zeros(1,3,64,64)).shape[1]
        # rpn module for proposal generation
        self.rpn = RPN_Module(cfg, bb_ch)
        # rcnn module for matching and refinement
        self.rcnn = RCNN_Module(cfg)
        
    def normalize_tensor(self, x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        # x: batch image tensor [bnum, 3, h, w]
        x[:,0],x[:,1],x[:,2] = (x[:,0]-mean[0])/std[0], (x[:,1]-mean[1])/std[1], (x[:,2]-mean[2])/std[2]
        return x
    
    def forward(self, x,y, xb, xfa=None, add_box=None):
        # x : query image, Tensor, [bnum, 3, img_h, img_w]
        # y : search image, Tensor, [bnum, 3, img_h, img_w]
        # xb : bbox coordinates for pos samples: list, [list_posbb] where len(list_bb)=bnum, list[i] = Tensor[1,4]
        # xfa : xfeats for feature reuse : tuple (xfa_tri, xfa_pos, xfa_neg)

        # pool pos/neg feats from x (if negft:true)
        xfa_in = self.get_feats_xfa(x, xb) if xfa is None else xfa
        # get feats from backbone (if not available)
        xf = self.backbone(self.normalize_tensor(x)) if xfa_in is None else None
        yf = self.backbone(self.normalize_tensor(y))
        # get roi proposals, pooled feats and rpn outputs
        rois, scores, rpn_feats, out_rpn = self.rpn(xf, yf, xb, xfa_in[0], add_box)
        # matching confidence scores and bbox refinement
        rcnn_feats = (xfa_in[1], xfa_in[2], rpn_feats[2]) #(pos_feat, neg_feat, can_feat)
        out_rcnn = self.rcnn(rcnn_feats, rois) #(cf,op,bb,roi)
        
        return out_rpn, out_rcnn
    
        
    def forward_box(self, x,y, xb, xfa=None, add_box=None, nbox=1):
        # params
        num_batch = y.shape[0]
        # get final outputs
        out_rpn, out_rcnn = self.forward(x,y, xb, xfa, add_box)
        out_cf, out_op, out_bb, out_br = out_rcnn
        # choose single box with max score for each batch - obtain scores, choose max score idxs for each batch
        # pos score + mean neg score
        out_ff_pos = torch.exp(out_cf[...,0,0]) / (torch.exp(out_cf[...,0,0])+torch.exp(out_cf[...,0,1]))
        out_ff_neg = torch.exp(out_cf[...,1:,1]) / (torch.exp(out_cf[...,1:,0])+torch.exp(out_cf[...,1:,1])) if self.head_negff else 1.
        # product of negative scores
        out_ff_neg = torch.prod(out_ff_neg, dim=-1) if self.head_negff else 1.  #torch.mean/torch.sum
        # overlap score
        out_op = torch.sigmoid(out_op[...,0]) if self.head_oproi else 1.
        
        # fianl score = pos_score*overlap_score*neg_score
        out_ff = out_ff_pos*out_ff_neg*out_op
        sort_idxs = out_ff.argsort(descending=True, dim=1)
        # returns bb coordinates for each batch
        out_bb_b = []
        out_ff_b = []
        for i in range(num_batch):
            out_bb_b.append(out_bb[i,sort_idxs[i,:nbox]]) # out_bb out_br
            out_ff_b.append(out_ff[i,sort_idxs[i,:nbox]])
        out_bb_b = torch.stack(out_bb_b)
        out_ff_b = torch.stack(out_ff_b)
        
        return out_bb_b, out_ff_b, (out_rpn, out_rcnn)
        

    def get_feats_xfa(self, x, xb):
        # params
        num_batch = x.shape[0]
        thres,nfeat = self.nft_param
        nbox_num, nbox_thr = self.rpn.boxes.bb_nums, self.rpn.boxes.bb_thres
        # change numof candidate negative boxes
        self.rpn.boxes.bb_nums, self.rpn.boxes.bb_thres = 64,0.5
        # get pos and neg feats from query img
        xf = self.backbone(self.normalize_tensor(x))
        # roi proposals and feats
        rois, scores, feats, _ = self.rpn(xf, xf, xb, add_box=xb, pool_xf=True)
        xfa_tri = feats[0]
        xfa_pos = feats[2][:,-1]
        yfa     = feats[2][:,:-1]
        # negative feature mining inside xf
        if self.head_negff:
            xfa_neg = torch.zeros(num_batch, nfeat, self.head_nfeat, self.roip_size, self.roip_size).cuda()
            for i in range(num_batch):
                # get ious per batch, choose feature idxs with lower iou < thres
                xb_i, roi_i, score_i = xb[i], rois[i][:-1,:], scores[i]
                iou_i = jaccard(xb_i, roi_i)[0]
                idx_sel = torch.nonzero( iou_i < thres )[:,0]
                idx_sel = idx_sel[:nfeat]
                # if numof features insufficient: repeat last idx
                if len(idx_sel)==0:
                    continue
                if len(idx_sel)<nfeat:
                    for _ in range(nfeat-len(idx_sel)):
                        idx_sel = torch.cat((idx_sel, idx_sel[[-1]]))
                xfa_neg[i] = yfa[i, idx_sel]
        else:
            xfa_neg = None
        # restore default box nums
        self.rpn.boxes.bb_nums, self.rpn.boxes.bb_thres = nbox_num, nbox_thr
        # return (xfa_tri, xfa_pos, xfa_neg)
        return (xfa_tri, xfa_pos, xfa_neg)
        
        

