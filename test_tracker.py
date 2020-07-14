import os,sys,argparse,time,cv2

import numpy as np
import matplotlib.pyplot as plt
from cfgs import cfg_test as cfg
import torch

from tracker import Tracker
# from tracker_batch import Tracker

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_db_name = 'lasot'
_save_txt = True
_calc_auc = True
_out_vid  = False 

    
def run_eval(idx=-1):
    tracker = Tracker(cfg=cfg, db_name=_db_name, idx=idx)
    tic = time.time()
    res, fps, auc = tracker.run_track_db(seq_list=None, save_res=_save_txt, calc_auc=_calc_auc, out_vid=_out_vid)

    if _calc_auc:
        res_str = 'db: '+ _db_name + ', auc: '+str(np.mean(auc))[:6]+ ', fps: '+str(np.mean(fps))[:5]+ ', ckpt: '+tracker.chkpt_file[5:-4] + '\n'
        with open('all_results.txt','a') as res_file:
            res_file.write(res_str)

    print 'elaptime ' + str((time.time()-tic)/60.)[:6] + ' mins'    
    return np.mean(auc)



run_eval()

