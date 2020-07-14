import torch
import torch.nn as nn
import torch.nn.functional as F
from focal_loss import FocalLoss
from utils import down2n
from box_utils import jaccard


class Track_Loss(nn.Module):
    def __init__(self, cfg):
        super(type(self), self).__init__()
        # params and flags
        self.loss_lambda = cfg.loss_lambda
        self.im_size = cfg.im_size
        self.map_size = (down2n(cfg.im_size[0],cfg.conv_npool[-1]),down2n(cfg.im_size[1],cfg.conv_npool[-1]))
        self.bbox_thres = cfg.bbox_thres
        self.head_oproi = cfg.head_oproi
        # loss objects
        self.cl_loss = FocalLoss(gamma=cfg.loss_gamma, alpha=cfg.loss_alpha, size_average=True)
        self.cf_loss = nn.CrossEntropyLoss()
        self.op_loss = nn.BCEWithLogitsLoss()
        
    
    def get_cl_loss(self, cl, gt):
        # return classification loss for rpn module, use focal loss
        # get positive instance indices from gt [bnum, map_h, map_w, 2]
        pos_idxs = gt.flatten().nonzero()[:,0]
        pos_nums = pos_idxs.shape[0]
        cl_f = cl.reshape(-1,2)
        gt_f = (1-gt).flatten().long()
        loss = self.cl_loss(cl_f, gt_f)
        # averaging
        #loss /= float(pos_nums) if pos_nums>0 else gt.numel()
        return loss
    
    
    def get_re_loss(self, re, gr, gt, eps=1e-7):
        # return box regression loss for positive instances
        # get positive instance indices from gt [bnum, map_h, map_w, 2]
        pos_idxs = gt.flatten().nonzero()[:,0]
        pos_nums = pos_idxs.shape[0]
        if pos_nums < 1:
            return 0.
        # select corresponding instances in regularization results
        gr_sel = gr.reshape(-1,4)[pos_idxs] # [pos_idxs, ltrb]
        re_sel = re.reshape(-1,4)[pos_idxs]
        
        # iou calculation - intersection
        iou_inter = torch.min(re_sel, gr_sel)
        iou_inter = (iou_inter[:,0]+iou_inter[:,2])*(iou_inter[:,1]+iou_inter[:,3]) # area = (l+r)*(t+b)
        # iou calculation - union
        gr_area = (gr_sel[:,0]+gr_sel[:,2])*(gr_sel[:,1]+gr_sel[:,3]) # area = (l+r)*(t+b)
        re_area = (re_sel[:,0]+re_sel[:,2])*(re_sel[:,1]+re_sel[:,3]) # area = (l+r)*(t+b)
        iou_union = gr_area + re_area - iou_inter + eps
        # iou calculation - inter / union
        iou_sel = (iou_inter+1.) / (iou_union+1.)
        # total iou loss
        loss = torch.mean(1.-iou_sel)
        return loss
        
    
    def get_rcnn_loss(self, cf, op, bb, br, gb):
        # cf = [numb, numbb, 1+nnum, 2(pn)] (output binary class)
        # op = [numb, numbb, 1] (output iou overlap score)
        # bb = [numb, numbb, 4(xyxy)] (output refined bbox)
        # br = [numb, numbb, 4(xyxy)] (output unrefined bbox)
        # gb = [numb, 4] (ground truth bbox)
        # sizes
        num_batch = cf.shape[0]
        num_boxes = cf.shape[1]
        num_negbb = cf.shape[2]-1
        # per batch iteration
        loss, total_pos = 0,0
        for i in range(num_batch):
            # find positive instances in a batch (bb overlap > threshold)
            cf_i = cf[i]                  # [numbox, 1+nnum, 2]
            op_i = op[i] if self.head_oproi else None   # [numbox, 1]
            bb_i = bb[i]                  # [numbox, 4] = [numbox, x0y0x1y1]
            br_i = br[i]                  # [numbox, 4] = [numbox, x0y0x1y1]
            gb_i = gb[i].unsqueeze(0)     # [1,4] = [1, x0y0x1y1]
            # iou for rois
            iou_br = jaccard(gb_i, br_i)[0]
            pos_idxs = (iou_br >=self.bbox_thres[0]).nonzero()[:,0]
            neg_idxs = (iou_br < self.bbox_thres[1]).nonzero()[:,0]
            pos_nums, neg_nums = pos_idxs.shape[0], neg_idxs.shape[0]
            total_pos += pos_nums
            
            # enforce iou overlap regression loss
            loss_op_i = self.op_loss(op_i[...,0][pos_idxs], iou_br[pos_idxs]) if (pos_nums>0) and (self.head_oproi) else 0.
            
            # enforce labels, binary cross entropy loss
            # pos input sample ~ pos/neg boxes
            cf_lbl_pos = torch.zeros(pos_nums, device=cf_i.device).long()
            cf_lbl_neg  = torch.ones(neg_nums, device=cf_i.device).long()
            loss_cf_i_pos_pos = self.cf_loss(cf_i[pos_idxs,0,:], cf_lbl_pos) if pos_nums>0 else 0.
            loss_cf_i_pos_neg = self.cf_loss(cf_i[neg_idxs,0,:], cf_lbl_neg) if neg_nums>0 else 0.
            loss_cf_i_pos = loss_cf_i_pos_pos + loss_cf_i_pos_neg
            
            # neg input sample ~ pos boxes
            if (num_negbb>0) and (pos_nums>0):
                cf_i_neg = cf_i[pos_idxs,1:,:].flatten(0,1) # [pos_nums*num_negbb, 2]
                cf_lbl_neg = torch.ones(pos_nums*num_negbb, device=cf_i.device).long()
                loss_cf_i_neg = self.cf_loss(cf_i_neg, cf_lbl_neg)
            else:
                loss_cf_i_neg = 0.
            loss_cf_i = loss_cf_i_pos + loss_cf_i_neg
            
            # iou for refined bb
            iou_bb = jaccard(gb_i, bb_i, eps=1.0)[0]
            # enforce box regression (only for positive instances), linear iou loss
            loss_bb_i = torch.mean(1. - iou_bb[pos_idxs]) if pos_nums>0 else 0
            # loss for single batch, add to total loss
            if pos_nums==0:
                loss_i = 0.
            else:
                loss_i = loss_cf_i + loss_bb_i + loss_op_i
            loss += loss_i
        
        # divide loss by batch size
        loss /= num_batch
        return loss, total_pos
            
    
    def forward(self, outs, gts, add_rcnn_loss=True):
        # parse network outputs
        out_rpn, out_rcnn = outs
        cl, re = out_rpn[0], out_rpn[1]
        cf, op, bb, br = out_rcnn[0], out_rcnn[1], out_rcnn[2], out_rcnn[3]
        # parse gts (gt_box, gt_cl, gt_re)
        gb, gt, gr = gts
        
        # loss for rpn outputs
        rpn_loss0 = self.get_cl_loss(cl, gt)
        rpn_loss1 = self.get_re_loss(re, gr, gt)
        rpn_loss = rpn_loss0 + rpn_loss1
        
        # loss for rcnn outputs
        rcnn_loss, total_pos = self.get_rcnn_loss(cf, op, bb, br, gb)
        
        # total loss
        if add_rcnn_loss:
            total_loss = rpn_loss + self.loss_lambda*rcnn_loss
        else:
            total_loss = rpn_loss
            
        return total_loss, [rpn_loss0, rpn_loss1, rcnn_loss, int(total_pos)]
        
        
        
        
        
        