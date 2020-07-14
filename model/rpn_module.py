import torch, time
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from utils import down2n
from fcos import FCOSHead
from boxes import BoxModule
from cbam import CBAModule
from context import ContextModule

# receives pairwise images as input, returns roi bounding box proposals and its pooled features
class RPN_Module(nn.Module):
    def __init__(self,cfg,bb_ch):
        super(type(self), self).__init__()
        # params
        self.im_size = cfg.im_size
        self.map_size = (down2n(cfg.im_size[0],cfg.conv_npool[-1]),down2n(cfg.im_size[1],cfg.conv_npool[-1]))
        self.scale_f = float(self.map_size[0]) / float(self.im_size[0])
        self.pool_size = cfg.roip_size
        self.head_dconv = cfg.head_dconv
        self.head_ctxff = cfg.head_ctxff
        # numof channels for backbone output, refined output
        self.bb_ch = bb_ch
        self.head_nfeat = cfg.head_nfeat
        # attetntion module and channel conversion for backbone outputs
        fmul = 3 if cfg.head_dconv else 1
        self.cbamod = CBAModule(self.head_nfeat*fmul)
        self.conv_x = nn.Conv2d(self.bb_ch, cfg.head_nfeat, 1)
        self.conv_y = nn.Conv2d(self.bb_ch, cfg.head_nfeat, 1)
        # detection head
        self.roi_head = FCOSHead(cfg)
        # nms box predictions
        self.boxes = BoxModule(cfg)
        # context module
        self.context_x = ContextModule(cfg) if self.head_ctxff[0] else None
        self.context_y = ContextModule(cfg) if self.head_ctxff[0] else None
        
        # init
        rpn_convs = nn.ModuleList([self.conv_x, self.conv_y])
        for m in rpn_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    
    def fmap_corr(self,xf,yf,pad=True,dilate=1):
        # get shapes
        xf_s = xf.shape # [bn, ch, xh, xw] (larger -> as image) [1, bn*ch, xh, xw]
        yf_s = yf.shape # [bn, ch, yh, yw] (smaller -> as filter) [bn*cn, 1, yh, yw]
        bn = xf_s[0];    cn = xf_s[1];
        # reshape dims
        xf_r = xf.view(1, bn*cn, xf_s[2], xf_s[3]) # [1, bn*cn, xh, xw]
        yf_r = yf.reshape(1, bn*cn, yf_s[2], yf_s[3]).transpose(0,1) # [bn*cn, 1, yh, yw] view
        # group conv [1, bn*cn, zh, zw] -> [bn, cn, zh, zw]
        if pad: pnum = (yf_s[-1] + (dilate-1)*(yf_s[-1]-1))//2
        else:   pnum = 0
        of = F.conv2d(input=xf_r, weight=yf_r, groups=bn*cn, bias=None, padding=pnum, dilation=dilate)
        of = of.view(bn, cn, of.shape[2], of.shape[3])
        return of
    
    
    def dconv_fmap_corr(self, yf, xf):
        if self.head_dconv:
            zf = []
            for i in range(len(xf)):
                zf.append(self.fmap_corr(yf, xf[i]))
            zf = torch.cat(zf, dim=1)
        else:
            zf = self.fmap_corr(yf, xf[1], pad=False)
        return zf
    
    
    def corr_head(self, xfa, yf):
        # cross corr for xfa
        zf = self.dconv_fmap_corr(yf,xfa)
        # attention module
        zf,at = self.cbamod(zf)
        # detection head
        cl,re,zf = self.roi_head(zf)
        # permute dims to [bnum, map_h, map_w, pred], where pred_cls=[neg/pos], pred_re=[ltrb distances]
        cl = cl.permute(0,2,3,1)
        re = re.permute(0,2,3,1)
        return zf,cl,re,at
        
    
    def pool_feat(self, xf, xb_p):
        # xb: list of boxes wrt each batch : list, where len(list)=bnum, list[i] = Tensor[N,4]
        # feats -> change channel nums [bnum, ndim, pool_sz, pool_sz]
        
        # original roi align
        xfa = [roi_align(xf, xb_p, (self.pool_size,self.pool_size), self.scale_f)]
        # additional feats
        if self.head_dconv:
            # d2
            psz = self.pool_size*2 -1
            xfa.append(roi_align(xf, xb_p, (psz,psz), self.scale_f))
            # p2
            psz = self.pool_size//2
            psz += 1 if psz%2==0 else 0
            xfa.append(roi_align(xf, xb_p, (psz,psz), self.scale_f))
        else:
            xfa.append(roi_align(xf, xb_p, (1,1), self.scale_f))
            
        return xfa
        
        
    def forward(self,xf_in,yf_in, xb, xfa_in=None, add_box=None, pool_xf=False):
        # xf,yf : Tensor, [bnum, ndim, map_size_h, map_size_w]
        # xb : list, [list_posbb] where len(list_xxxbb)=bnum, list_xxxbb[i] = Tensor[N,4]
        # xfa_in : trident feat pooled from initial xf for reuse
        # add_box : list of boxes to add roi list(add_box)=bnum, add_box[i] = Tensor[M,4]
        # pool_xf : pool-align feat from xf rather than yf

        # change channel num of input feature
        xf = self.conv_x(xf_in) if xfa_in is None else None
        yf = self.conv_y(yf_in) 
        # roi_align pooling from xf according to xb coordinates
        # use given feature if pooled feat xfa is already given
        xfa_tri = self.pool_feat(xf, xb) if xfa_in is None else xfa_in
        
        # fmap cross correlation + detection head = class, regression maps
        zf,cl,re,at = self.corr_head(xfa_tri, yf)
        pred_maps = (cl,re,at)
        
        # ==== obtain ROI bounding boxes and pooled features        
        # nms stage for box predictions : bboxes+scores
        pred_roibb, pred_roisc = self.boxes(cl,re)
        # add previous box (if exists)
        if add_box is not None:
            for bi in range(len(pred_roibb)):
                pred_roibb[bi] = torch.cat((pred_roibb[bi], add_box[bi]),dim=0)
                
        # pool feats for given boxes yf with shapes: yfa = [bnum, bbnum, cnum, pool_size, pool_size]
        num_boxes = self.boxes.bb_nums if add_box is None else self.boxes.bb_nums+add_box[0].shape[0]
        yf = xf if pool_xf else yf  # for initial frame feature fetching purposes
        yfa = roi_align(yf, pred_roibb, (self.pool_size,self.pool_size), self.scale_f)
        yfa = yfa.view(yf.shape[0], num_boxes, yf.shape[1], self.pool_size, self.pool_size)
        
        # (if specified) embed context feature into ROI features (yfa) based on box predictions (cl,re)
        if self.head_ctxff[0]:
            yfa = self.context_y(cl,re, zf,yfa,pred_roibb) if not pool_xf else self.context_x(cl,re, zf,yfa,pred_roibb)
        
        # feats = (xfa_tri, xfa_pos, yfa)
        pred_feats = (xfa_tri, xfa_tri[0], yfa)
        
        return pred_roibb, pred_roisc, pred_feats, pred_maps
        
        
        
        
        
        
        
    
    