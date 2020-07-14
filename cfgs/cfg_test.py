from cfg_res50 import *


# ===== PATH variables =====
# checkpoint path + experiment number
CHKPT_PATH = 'ckpt/'
CHKPT_CODE = 'res50'


# construct dataset info dict
db_info = dict()
# test sets
db_info['lasot'] = {'size': 280,
                    'path' : '/home/jhchoi/datasets5/LaSOTBenchmark/',
                    'dict' : 'dict/lasot_dict_test.npy'}

