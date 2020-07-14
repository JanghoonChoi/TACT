import numpy as np
import matplotlib.pyplot as plt
import cv2,time


def get_dtstr(sec=True):
    tst = time.localtime()
    if sec:
        outstr = str(tst.tm_year)[-2:] + str('%02d' % tst.tm_mon) + str('%02d' % tst.tm_mday) + str('%02d' % tst.tm_hour)+ str('%02d' % tst.tm_min)+ str('%02d' % tst.tm_sec)
    else:
        outstr = str(tst.tm_year)[-2:] + str('%02d' % tst.tm_mon) + str('%02d' % tst.tm_mday) + str('%02d' % tst.tm_hour)+ str('%02d' % tst.tm_min)
    return outstr

def imread_to_rgb(path):
    img_in = np.flip(cv2.imread(path, flags=cv2.IMREAD_COLOR), 2)/255.
    return img_in

def crop_img(I, x, y, w, h, center=False, mfill=False):
    im_h = I.shape[0]
    im_w = I.shape[1]
    
    if center:
        w0 = w // 2;    w1 = w - w0    # w = w0+w1
        h0 = h // 2;    h1 = h - h0    # h = h0+h1

        x_min = x - w0;    x_max = x+w1-1;
        y_min = y - h0;    y_max = y+h1-1;
    else:
        x_min = x;    x_max = x+w-1;
        y_min = y;    y_max = y+h-1;
    
    pad_l = 0;    pad_r = 0;
    pad_u = 0;    pad_d = 0;
    
    # bounds
    if x_min < 0:          pad_l = -x_min;            x_min = 0;
    if x_max > im_w-1:     pad_r = x_max-(im_w-1);    x_max = im_w-1;
    if y_min < 0:          pad_u = -y_min;            y_min = 0;
    if y_max > im_h-1:     pad_d = y_max-(im_h-1);    y_max = im_h-1;

    # crop & append
    J = I[y_min:y_max+1, x_min:x_max+1, :]
    
    # 0 size errors
    if J.shape[0] == 0 or J.shape[1] == 0:
        plt.imsave('crop_error_'+time.strftime('%y%m%d_%H%M%S',time.localtime())+'.png', I)
        print 'i: ',I.shape, (x,y,w,h),J.shape
        print 'i: ',(y_min,y_max+1),(x_min,x_max+1)
        # return black image for zero-dim images
        return np.zeros([h,w,3])
    
    if mfill:
        rsel = np.linspace(0, J.shape[0], 8, endpoint=False, dtype=int)
        csel = np.linspace(0, J.shape[1], 8, endpoint=False, dtype=int)
        fill = np.mean(J[rsel][:,csel], axis=(0,1))
    else:
        fill = (0,0,0)
    J = cv2.copyMakeBorder(J, pad_u,pad_d,pad_l,pad_r, cv2.BORDER_CONSTANT, value=fill)
    return J


def draw_bb_img(img0, x_min,y_min,width,height, color, stroke):
    img = img0.copy()
    img_h = img.shape[0]; img_w = img.shape[1];

    x_rng = np.array(range(width)) + x_min
    y_rng = np.array(range(height))+ y_min
    
    x_rng[x_rng> img_w-1-stroke] = img_w-1-stroke
    y_rng[y_rng> img_h-1-stroke] = img_h-1-stroke
    
    x_max = np.max(x_rng)
    y_max = np.max(y_rng)
    
    img[y_min:y_min+stroke][:, x_rng, :] = color # up
    img[y_max-stroke:y_max][:, x_rng, :] = color # down
    img[:, x_min:x_min+stroke, :][y_rng] = color # left
    img[:, x_max-stroke:x_max, :][y_rng] = color # right
    
    return img


def dist_succ(v_pred, v_gt, batch_size):
    maxvals = v_pred.max(axis=1).max(axis=1)
    v_gt_mod = v_gt.copy() + 1.
    
    idxs = list();   gt_idxs = list();
    for b_i in range(batch_size):
        maxpos = np.where(v_pred == maxvals[b_i])[1:3]
        if np.shape(maxpos)[1] > 1:
            maxpos = (np.array([maxpos[0][0]]), np.array([maxpos[1][0]]))
        idxs.append(maxpos)
        gt_idxs.append(center_of_mass(v_gt_mod[b_i]))
        
    idxs = np.array(idxs).reshape([batch_size, 2]).astype(float)
    gt_idxs = np.array(gt_idxs).reshape([batch_size, 2])
    
    dist = np.sum( ( idxs - gt_idxs )**2, axis=1 )
    dist = np.sqrt( dist )
    succ = (dist <= np.sqrt(2.))

    return dist, succ
    

def down2n(x, n):
    # returns input length of x after n-times of pooling/strides of 2
    if n == 1:
        return np.ceil(x/2.).astype(int)
    else:
        return down2n(np.ceil(x/2.), n-1).astype(int)


def gray2jet(I):
    # convert input gray image I to jet colormap image J
    # trapezoid func map [0,1]->[0,1] (rise:t0~t1, down:t2~t3)
    def tpz(xin, t0,t1,t2,t3):
        x = xin.copy()
        x[xin<=t0] = 0.
        x[(xin>t0)*(xin<=t1)] = (xin[(xin>t0)*(xin<=t1)] - t0) / (t1-t0)
        x[(xin>t1)*(xin<=t2)] = 1.
        x[(xin>t2)*(xin<=t3)] = (xin[(xin>t2)*(xin<=t3)] - t3) / (t2-t3)
        x[xin>t3] = 0.
        return x
    
    # respective rgb channel mappings
    J_r = tpz(I, 0.375, 0.625, 0.875, 1.125)
    J_g = tpz(I, 0.125, 0.375, 0.625, 0.875)
    J_b = tpz(I, -0.125, 0.125, 0.375, 0.625)
    
    J = np.zeros([I.shape[0], I.shape[1], 3])
    J[:,:,0] = J_r
    J[:,:,1] = J_g
    J[:,:,2] = J_b
    return J
    
