import os,sys,time,cv2

import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpe

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.build_model import Track_Model
from utils import crop_img, imread_to_rgb
from track_utils import box_overlap_score, result_curve

_batch_size = 8
_num_thread = 2

# tracking dataset
class Seq_dataset(Dataset):
    def __init__(self, cfg, seq_path, seq_imlist):
        self.cfg = cfg
        self.seq_path = seq_path
        self.seq_imlist = seq_imlist
        self.len = len(seq_imlist)-1
    
    def __len__(self):
        return self.len
    
    def read_img_resize(self, imf):
        img_orig = imread_to_rgb(imf)
        h_orig, w_orig, _ = img_orig.shape
        MAX_H, MAX_W = self.cfg.im_size
        s_f = float(MAX_H) / float(h_orig)
        if float(w_orig)*s_f > MAX_W:
            s_f = float(MAX_W) / float(w_orig)
        img_mod = cv2.resize(img_orig, (int(w_orig*s_f), int(h_orig*s_f)) )
        h_mod, w_mod, _ = img_mod.shape
        img_zero = np.zeros([MAX_H, MAX_W, 3])
        img_zero[:h_mod, :w_mod, :] = img_mod
        return img_zero, s_f
    
    def __getitem__(self, idx):
        seq_imlist = self.seq_imlist[1:]
        im_path = os.path.join(self.seq_path, seq_imlist[idx])
        im_frame,_ = self.read_img_resize(im_path)
        out_im = torch.Tensor(im_frame).permute(2,0,1)
        return out_im


# tracker object
class Tracker(nn.Module):
    def __init__(self, cfg, db_name=None, idx=-1):
        super(type(self), self).__init__()
        # model object
        cfg.batch_size = _batch_size
        self.net = Track_Model(cfg).cuda()
        self.net.eval()
        self.cfg = cfg
        # tracking db placeholders
        self.track_dbnm = None
        self.track_path = None
        self.track_dict = None
        # load model weights
        self.chkpt_file = [ckpt for ckpt in sorted(os.listdir(cfg.CHKPT_PATH)) if ckpt.find(cfg.CHKPT_CODE)>0][idx]
        ckpt = torch.load(cfg.CHKPT_PATH+self.chkpt_file)
        self.net.load_state_dict(ckpt['model_state_dict'], strict=False)
        print 'ckpt: ' + self.chkpt_file
        # load db
        if db_name is not None:
            self.load_track_db(db_name)
    
    
    def load_track_db(self, name):
        # load dataset
        self.track_dbnm = name
        self.track_path = self.cfg.db_info[name]['path']
        self.track_dict = np.load(self.cfg.db_info[name]['dict'], allow_pickle=True).item()
        print 'dataset: ' + name
    
    
    def read_img_resize(self, imf):
        img_orig = imread_to_rgb(imf)
        h_orig, w_orig, _ = img_orig.shape
        MAX_H, MAX_W = self.cfg.im_size
        s_f = float(MAX_H) / float(h_orig)
        if float(w_orig)*s_f > MAX_W:
            s_f = float(MAX_W) / float(w_orig)
        img_mod = cv2.resize(img_orig, (int(w_orig*s_f), int(h_orig*s_f)) )
        h_mod, w_mod, _ = img_mod.shape
        img_zero = np.zeros([MAX_H, MAX_W, 3])
        img_zero[:h_mod, :w_mod, :] = img_mod
        return img_zero, s_f

    
    def run_track_seq(self, seq_name, seq_path, seq_imlist, seq_gt, save_res=False):
        # preliminary
        if ['got10k', 'trackingnet', 'uav123', 'uav20l', 'nuspro'].count(self.track_dbnm) == 0:
            seq_path = os.path.join(seq_path, 'img/')
        seq_path = os.path.join(self.track_path, seq_path)
        # results placeholder
        seq_len = len(seq_imlist)
        seq_res, seq_fps = [],[]
        # seq db
        seq_tdb = Seq_dataset(self.cfg, seq_path, seq_imlist)
        seq_tdl = DataLoader(seq_tdb, batch_size=self.cfg.batch_size, num_workers=_num_thread)
        # initial frame
        i = 0
        # init state = [xmin, ymin , width, height]
        state = seq_gt[0,:].copy().astype(float)
        seq_res.append(np.expand_dims(state.copy(),0))
        # init frame
        im_frame, s_f = self.read_img_resize(os.path.join(seq_path, seq_imlist[0]))
        # convert state to [xmin, ymin, xmax, ymax]*scale_factor
        state_mod = np.array([state[0], state[1], state[0]+state[2], state[1]+state[3]])*s_f
        # init feats
        net_im = torch.Tensor(im_frame).unsqueeze(0).permute(0,3,1,2).repeat_interleave(self.cfg.batch_size,0).cuda()
        net_bb = [torch.Tensor(state_mod).unsqueeze(0).cuda()]*self.cfg.batch_size
        with torch.no_grad():
            xfa = self.net.get_feats_xfa(net_im, net_bb)
        
        # tracking part
        for i, im_frame in enumerate(seq_tdl):
            sys.stdout.write("\r"+str((i)*self.cfg.batch_size)+'/'+str(seq_len))
            # subsequent frames
            # read img
            tic = time.time()
            temp_sz = im_frame.shape[0]
            net_im = torch.zeros(self.cfg.batch_size, 3, self.cfg.im_size[0], self.cfg.im_size[1])
            net_im[:temp_sz] = im_frame
            net_im = net_im.cuda()
            # find target
            with torch.no_grad():
                net_out_bb, _, _ = self.net.forward_box(None,net_im, None, xfa=xfa, nbox=1)
            state_mod = net_out_bb.squeeze().detach().cpu().numpy() / s_f
            state = np.zeros_like(state_mod)
            state[:,0], state[:,1], state[:,2], state[:,3] = state_mod[:,0], state_mod[:,1], state_mod[:,2]-state_mod[:,0], state_mod[:,3]-state_mod[:,1]
            # store results
            seq_res.append(state.copy())
            seq_fps.append((time.time()-tic))
        
        # concat dims
        seq_res = np.concatenate(seq_res)[:seq_len]
        seq_fps = 1./(np.sum(seq_fps)/float(seq_len))
        # save res
        if save_res:
            
            if self.track_dbnm == 'got10k':
                os.mkdir('output/'+seq_name)
                np.savetxt('output/'+seq_name+'/'+seq_name+'_001.txt', seq_res, fmt='%.4f', delimiter=',')
            else:
                np.savetxt('output/'+seq_name+'.txt', seq_res, fmt='%.4f', delimiter=',')

        return seq_res, seq_fps
    
    
    def run_track_db(self, seq_list=None, out_vid=False, calc_auc=True, save_res=False):
        # results placeholder
        db_res = dict()
        db_fps = []
        db_auc = []
        db_suc = []
        # per-sequence operation
        seq_list = self.track_dict.keys() if seq_list is None else seq_list
        seq_nums = len(seq_list)
        for s_i, seq in enumerate(seq_list):
            # print seq name
            print '('+ str(s_i+1) +'/' + str(seq_nums) + '):' + seq
            # seq path+imlist+gt
            seq_dict = self.track_dict[seq]
            seq_path = seq if not seq_dict.has_key('path') else seq_dict['path']
            seq_imlist = seq_dict['img']
            seq_gt = seq_dict['gt']
            # run tracking
            seq_res, seq_fps = self.run_track_seq(seq, seq_path, seq_imlist, seq_gt, save_res=save_res)
            db_res[seq] = seq_res
            db_fps.append(seq_fps.mean())
            # calc and display auc 
            if calc_auc:
                seq_iou = box_overlap_score(seq_res, self.track_dict[seq]['gt'])
                seq_suc = seq_iou>0.5
                seq_auc = result_curve(seq_iou, 21)
                db_auc.append(seq_auc)
                db_suc.append(seq_suc)
                print ', fps: ' + str(seq_fps.mean())[:6],
                print ', suc: ' + str(float(np.sum(seq_suc))/seq_res.shape[0])[:6],
                print ', auc: ' + str(np.mean(seq_auc))[:6] + ', mean_auc: ' + str(np.mean(db_auc))[:6]
            if out_vid:
                self.draw_vid_seq(seq_res, seq)
            
        # display overall results
        if calc_auc:
            print '\nmean fps: ' + str(np.mean(db_fps))[:6]
            print 'mean suc: ' + str(np.mean(np.concatenate(db_suc)))[:6]
            print 'mean auc: ' + str(np.mean(db_auc))[:6]
        
        return db_res, db_fps, db_auc
            
    
    def draw_vid_seq(self, seq_res, seq_name):
        print '> make video seq...',
        # preliminaries
        seq_dict = self.track_dict[seq_name]
        seq_path = seq_name if not seq_dict.has_key('path') else seq_dict['path']
        if self.track_dbnm is not 'got10k':
            seq_path = os.path.join(seq_path, 'img/')
        seq_path = os.path.join(self.track_path, seq_path)
        seq_len = len(seq_dict['img'])
        # draw for all frames
        im_slist = []
        for i, imf in enumerate(seq_dict['img']):
            # read img
            im_frame = imread_to_rgb(os.path.join(seq_path,imf))
            # draw bb = [xmin, ymin, width, height]
            bb = seq_res[i].astype(int)
            im_frame = cv2.rectangle(im_frame, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (1,0,0), 3)
            # fnum text
            fnum_str = str('%04d'%i)
            im_frame = cv2.putText(im_frame, fnum_str, (0,im_frame.shape[0]), cv2.FONT_HERSHEY_DUPLEX, im_frame.shape[0]/350., (1,1,0))
            # save img
            im_sname = os.path.join('.temp/', seq_name +'_'+ fnum_str + '.jpg')
            im_slist.append(im_sname)
            plt.imsave(im_sname, im_frame)
        
        # encode video
        vid_clip = mpe.ImageSequenceClip(im_slist, fps=30)
        vid_clip.write_videofile('test.mp4', logger=None)
        print 'done'
        return
        

    def clean_temp_dir(self, temp_dir='.temp/'):
        flist = os.listdir(temp_dir)
        for f in flist:
            os.remove(os.path.join(temp_dir, f))
        print '> cleaned cache folder'
        return
            
        