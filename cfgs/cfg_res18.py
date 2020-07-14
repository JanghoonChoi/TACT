# define training flags
EXP_COMMENT = 'res18_final_model'

# define training parameters
im_size          = (400, 666)          # image max sizes (height, width)
batch_size       = 4                   # batch size for training
batch_size_val   = 8                   # batch size for validation

name_bbnet       = 'resnet18'          # choose backbone : [resnet18, resnet34, resnet50, wide_resnet50_2, resnext50_32x4d]
conv_npool       = (2,3,4,4)           # numof pooling for each output for backbone network (default:[2,3,4,5])
roip_size        = 5                   # spatial sizeof roi-aligned features (default:7x7)
head_nconv       = 2                   # numof conv layers for detection heads
head_nfeat       = 256                 # channel dim. for feature maps in detection heads
head_nlocl       = True                # use or not use nonlocal layer (embedded gaussian)
head_dconv       = True                # use or not use dilated convs
head_negff       = False               # use or not use negative feats for final scoring
head_oproi       = False               # use or not use roi overlap prediction branch
head_ctxff       = (True, 3)           # use or not use context feature fusion + fusion scheme number (0:cat,1:add,2:cbam,3:film)
bbox_thres       = (0.5, 0.4)          # bbox thresholds for pos/neg samples for training
nms_param        = (0.90, 64)          # nms params (overlap_threshold_pos, _neg, num_candidate_boxes)
nft_param        = (0.4, 6)            # negative feat param (overlap_threshold, num_negative_boxes)

num_epochs       = int(1e+3)           # numof training epochs
training_iter    = int(1e+5)           # numof training iterations per epoch
lr_start         = 1e-4                # learning rate (initial)
lr_decay         = 0.50                # learning rate decay rate per loop
lr_decay_step    = 2000000             # learning rate decay steps
w_decay          = 1e-5                # weight decay rate for optimizer
loss_lambda      = 1.00                # balancing term for loss function (cls + lambda*reg)
loss_gamma       = 2.00                # focal loss gamma value (penalty on easy examples)
loss_alpha       = None                # focal loss alpha value (pos/neg example balancing)


# ===== PATH variables =====
# checkpoint/init path + experiment number
CHKPT_PATH, INITP_PATH = 'ckpt/', 'init/init_res18_weights.tar'
CHKPT_CODE = ''
# validation set dump path
VALID_PATH = '/home/jhchoi/datasets3/track_valid_set_fcos.npz'
    



