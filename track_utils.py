import numpy as np
import cv2
from utils import crop_img
import matplotlib.pyplot as plt


def result_curve(result, num_points):
    #num_seqs = 1 #len(result)
    all_seq_plot = np.zeros(num_points)
    
    #for i in range(num_seqs):
    seq_iou = result
    seq_len = np.size(result)
    seq_plot = list()

    bb_thvars = np.linspace(0,1,num_points)
    for bbth in bb_thvars:
        ratio_th = np.sum(seq_iou > bbth).astype(float) / seq_len
        seq_plot.append(ratio_th)
        
    return np.array(seq_plot)


def result_curve_px(result, num_points):
    num_seqs = len(result)
    all_ious = np.array([])
    
    for i in range(num_seqs):
        all_ious = np.append(all_ious, result[i])
    
    num_frames = len(all_ious)
    bb_thvars = np.linspace(0,50,num_points)
    all_ratio_th = np.array([])
    
    for bbth in bb_thvars:
        ratio_th = np.sum(all_ious <= bbth).astype(float) / num_frames
        all_ratio_th = np.append(all_ratio_th, ratio_th)
    
    return all_ratio_th


def box_overlap_area(A,B):
    if A.ndim == 1:
        A_xmin = A[0]; A_xmax = A_xmin+A[2]; A_ymin = A[1]; A_ymax = A_ymin+A[3]
        B_xmin = B[0]; B_xmax = B_xmin+B[2]; B_ymin = B[1]; B_ymax = B_ymin+B[3]
        # x,y dim overlap?
        x_over = max(0, min(A_xmax,B_xmax)-max(A_xmin,B_xmin))
        y_over = max(0, min(A_ymax,B_ymax)-max(A_ymin,B_ymin))
        # area of overlap
        area_overlap = x_over*y_over
        return area_overlap
    else:
        num_d  = A.shape[0]
        A_xmin = A[:,0]; A_xmax = A_xmin+A[:,2]; A_ymin = A[:,1]; A_ymax = A_ymin+A[:,3]
        B_xmin = B[:,0]; B_xmax = B_xmin+B[:,2]; B_ymin = B[:,1]; B_ymax = B_ymin+B[:,3]
        # x,y dim overlap?
        x_over = np.max([np.zeros(num_d), np.min([A_xmax,B_xmax], axis=0)-np.max([A_xmin,B_xmin], axis=0)], axis=0)
        y_over = np.max([np.zeros(num_d), np.min([A_ymax,B_ymax], axis=0)-np.max([A_ymin,B_ymin], axis=0)], axis=0)
        # area of overlap
        area_overlap = x_over*y_over
        return area_overlap


def box_overlap_score(A,B):
    if A.ndim == 1:
        A_width = A[2]; A_height = A[3]; B_width = B[2]; B_height = B[3];
        A_area = A[2]*A[3]; B_area = B[2]*B[3];
        area_overlap = box_overlap_area(A,B)
        area_union = A_area + B_area - area_overlap
        return area_overlap / area_union
    else:
        A_width = A[:,2]; A_height = A[:,3]; B_width = B[:,2]; B_height = B[:,3];
        A_area = A[:,2]*A[:,3]; B_area = B[:,2]*B[:,3];
        area_overlap = box_overlap_area(A,B)
        area_union = A_area + B_area - area_overlap
        return area_overlap / area_union

