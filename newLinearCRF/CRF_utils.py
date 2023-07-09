import math
import numpy as np
from tqdm import tqdm
import os
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import time

import CONSTANT
DEVICE = torch.device("cuda")
CPU = torch.device("cpu")
DEVICE=CPU
def gaussian_kernel(size = 5, sigma = 2):
    # Create an (size x size) grid of coordinates
    x, y = np.mgrid[-(size // 2):(size // 2) + 1, -(size // 2):(size // 2) + 1]
    # Calculate the 2D Gaussian function at each coordinate
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    # Normalize the kernel so that the sum of its elements is 1
    kernel /= kernel.sum()
    return kernel

KERNEL = gaussian_kernel()
KERNEL = torch.from_numpy(KERNEL).to(DEVICE)

#######################
# Model Parameters
#######################
WIDTH = CONSTANT.WIDTH # 238
HEIGHT = CONSTANT.HEIGHT # 168
NEIGHBOR_SHIFT = CONSTANT.NEIGHBOR_SHIFT  # 2
NEIGHBOR_LENGTH = NEIGHBOR_SHIFT * 2 + 1  # 5
NEIGHBOR = NEIGHBOR_LENGTH * NEIGHBOR_LENGTH  # 25
SEQ_LENGTH = CONSTANT.SEQ_LENGTH # 10000

#######################
# Hyper Parameters
#######################
WEIGHT_TRANSITION = CONSTANT.WEIGHT_TRANSITION # -9.0
WEIGHT_UNARY = CONSTANT.WEIGHT_UNARY # -0.3
WEIGHT_HEADING = CONSTANT.WEIGHT_HEADING # 0.8
WEIGHT_DISTANCE = CONSTANT.WEIGHT_DISTANCE # -0.6
WEIGHT_LOC = CONSTANT.WEIGHT_LOC # 20.0
BIAS_DISTANCE = CONSTANT.BIAS_DISTANCE # 1.0
HEADING_FILLNA = CONSTANT.HEADING_FILLNA # 0.0  # what value to fill NaN values in heading scores

def localization_score(localizations):
    loc_score = torch.zeros([HEIGHT, WIDTH], device=DEVICE)
    for (y, x) in localizations:
        loc_score[y-3:y+2, x-3:x+2] = KERNEL
    loc_score = loc_score.unsqueeze(2).repeat(1, 1, NEIGHBOR)
    # print(loc_score.shape)
    return loc_score

def viterbi_reverse(score):
    # shifted = np.zeros([HEIGHT + 2 * NEIGHBOR_SHIFT, WIDTH + 2 * NEIGHBOR_SHIFT, NEIGHBOR])
    # shifted = np.full((HEIGHT + 2 * NEIGHBOR_SHIFT, WIDTH + 2 * NEIGHBOR_SHIFT, NEIGHBOR), np.NINF)
    shifted = torch.full(
        (HEIGHT + 2 * NEIGHBOR_SHIFT, WIDTH + 2 * NEIGHBOR_SHIFT, NEIGHBOR),
        float("-inf"),
        device=DEVICE,
    )
    for idx in range(NEIGHBOR):
        y, x = np.unravel_index(idx, (NEIGHBOR_LENGTH, NEIGHBOR_LENGTH))
        shifted[y : HEIGHT + y, x : WIDTH + x, idx] = score[:, :, idx]
    shifted_flip = torch.flip(
        shifted, dims=[2]
    )  # NOTE: the shifted matrices are in reverse order
    score_reverse = shifted_flip[
        NEIGHBOR_SHIFT : HEIGHT + NEIGHBOR_SHIFT,
        NEIGHBOR_SHIFT : WIDTH + NEIGHBOR_SHIFT,
        :,
    ]
    # for x in range(NEIGHBOR_LENGTH):
    #     for y in range(NEIGHBOR_LENGTH):
    #         idx = np.ravel_multi_index(np.array([y, x]), (NEIGHBOR_LENGTH, NEIGHBOR_LENGTH))
    #         idx = NEIGHBOR_LENGTH * y + x
    #         shifted[y:HEIGHT + y, x:WIDTH + x, idx] = score[:, :, idx]
    return score_reverse

def getAngle(vector_1,vector_2):
    zero = [0,0]
    if(all(vector_1) ==zero):
        return 0
    if(all(vector_2) ==zero):
        return 0
    norm_1 = np.linalg.norm(vector_1)
    norm_2 = np.linalg.norm(vector_2)

    if norm_1 == 0 or norm_2 == 0:
        return 0

    unit_vector_1 = vector_1 / norm_1
    unit_vector_2 = vector_2 / norm_2

    dot_product = np.dot(unit_vector_1, unit_vector_2)
    # Handle dot product outside the valid range [-1, 1]
    if dot_product >= -1.0 and dot_product <= 1.0:
        angle = np.arccos(dot_product)
    elif dot_product > 1.0:
        angle = 0.0
    elif dot_product < -1.0:
        angle = np.pi
    else:
        # Handle other cases if necessary
        angle = 0.0

    return angle

def rotate (vector,angle):
    # print(vector)
    # print(angle)
    vx = vector[0]
    vy = vector[1]
    x = vector[0] * math.cos(angle) + vector[1] * math.sin(angle)
    y = vector[0] * math.sin(angle) + vector[1] * math.cos(angle)

    new = []
    new.append(x)
    new.append(y)
    newNP = np.array(new).astype(int)
    # print(newNP)
    return newNP

def project(a, b):
    a = np.float64(a)
    b = np.float64(b)
    proj = np.dot(a, b) / np.linalg.norm(b)**2 * b
    return proj
  
def covertToVectorList(S):
    ret = []
    for i in range(1,len(S)):
        x = S[i] [0] - S[i-1] [0]
        y = S[i] [1] - S[i-1] [1]
        ret.append((x,y))
    return ret

def correctAngle(S,Z):

    S_vec = covertToVectorList(S)
    Z_vec = covertToVectorList(Z)
    sum = 0
    for i in range(1,len(S_vec)):
        x1 = S_vec[i] [0] - S_vec[i-1] [0]
        y1 = S_vec[i] [1] - S_vec[i-1] [1]

        x2 = Z_vec[i] [0] 
        y2 = Z_vec[i] [1] 

        a = getAngle([x1,y1],[x2,y2])
        sum += a
    avg = sum/len(S_vec)
    # print(Z)
    # print("")
    # print(S)
    # Slen = len(S)
    # sum = 0
    # for i in range(1,Slen):
    #     x1 = S[i] [0] - S[i-1] [0]
    #     y1 = S[i] [1] - S[i-1] [1]
    #     x2 = Z[i] [0] - Z[i-1] [0]
    #     y2 = Z[i] [1] - Z[i-1] [1]
    #     # x2 = Z[i] [0]
    #     # y2 = Z[1] [1]
    #     a = getAngle([x1,y1],[x2,y2])
    #     sum += a
    
    
    # print(avg)
    return avg

# position old: inertial old
# position new: inertial new
# score_last_step: score matrix
# score_precalculate: score map matrix
# localizations: wifi localization
# vec_s_list: estimate window trace
# vec_z_lsit: inertial window list
def score_loc2(position_old, position_new, score_last_step, score_precalculate, localizations,vec_s_list,vec_z_list):
    
    position_diff = position_new - position_old 
    angle = correctAngle(vec_s_list,vec_z_list)
    position_rotate = rotate(position_diff,angle)
    position_delta = position_rotate + np.array([2, 2]) 

    idx = np.ravel_multi_index(position_delta, [5, 5])

    # score_last_step = np.repeat(score_last_step[:,:,np.newaxis], NEIGHBOR, axis=2)
    score_last_step = score_last_step.unsqueeze(2).repeat(1, 1, NEIGHBOR)
    loc_score = localization_score(localizations)
    score_all = score_precalculate[idx] + score_last_step + loc_score * WEIGHT_LOC 

    # print("viterbi reverse")
    score_viterbi = viterbi_reverse(score_all)  # shape (HEIGHT, WIDTH, NEIGHBOR)
    # print("find max")
    score_this_step, score_traceback = torch.max(
        score_viterbi, dim=2
    )  # shape (HEIGHT, WIDTH), value is float score
    # score_traceback = torch.argmax(score_viterbi, dim=2)  # shape (HEIGHT, WIDTH), value from 0 to 24
    return score_this_step, score_traceback

# vec_s_list: trace estimated position in window
# vec_z_list: window lenghth inertial trajectory list 
def score2(position_old, position_new, score_last_step, score_precalculate,vec_s_list,vec_z_list):
    
    position_diff = position_new - position_old 
    angle = correctAngle(vec_s_list,vec_z_list)
    position_rotate = rotate(position_diff,angle)

    position_delta = position_rotate + np.array([2, 2]) 


    # access reverse
    # get the index of table by the ravel_multi_index -- go check the ravel_multi_index def
    idx = np.ravel_multi_index(position_delta, [5, 5])
    # score_last_step = np.repeat(score_last_step[:,:,np.newaxis], NEIGHBOR, axis=2)
    score_last_step = score_last_step.unsqueeze(2).repeat(1, 1, NEIGHBOR)
    score_all = score_precalculate[idx] + score_last_step
    # print("viterbi reverse")
    score_viterbi = viterbi_reverse(score_all)  # shape (HEIGHT, WIDTH, NEIGHBOR)
    # print("find max")
    score_this_step, score_traceback = torch.max(
        score_viterbi, dim=2
    )  # shape (HEIGHT, WIDTH), value is float score
    # score_traceback = torch.argmax(score_viterbi, dim=2)  # shape (HEIGHT, WIDTH), value from 0 to 24
    return score_this_step, score_traceback

def score_loc(position_old, position_new, score_last_step, score_precalculate, localizations):
    position_delta = position_new - position_old + np.array([2, 2])

    idx = np.ravel_multi_index(position_delta, [5, 5])
    # score_last_step = np.repeat(score_last_step[:,:,np.newaxis], NEIGHBOR, axis=2)
    score_last_step = score_last_step.unsqueeze(2).repeat(1, 1, NEIGHBOR)
    loc_score = localization_score(localizations)
    score_all = score_precalculate[idx] + score_last_step + loc_score * WEIGHT_LOC 

    # # print(loc_score.shape)
    # angle_score = torch.zeros([HEIGHT, WIDTH], device=DEVICE)
    # for (y,x) in localizations:
    #     angle_score[y-3:y+2, x-3:x+2] = getAngle(position_delta,(x,y))/5 * -1
    # angle_score = angle_score.unsqueeze(2).repeat(1, 1, NEIGHBOR)
    # score_all = score_all + angle_score
    
    # print("viterbi reverse")
    score_viterbi = viterbi_reverse(score_all)  # shape (HEIGHT, WIDTH, NEIGHBOR)
    # print("find max")
    score_this_step, score_traceback = torch.max(
        score_viterbi, dim=2
    )  # shape (HEIGHT, WIDTH), value is float score
    # score_traceback = torch.argmax(score_viterbi, dim=2)  # shape (HEIGHT, WIDTH), value from 0 to 24
    return score_this_step, score_traceback


def score(position_old, position_new, score_last_step, score_precalculate):
    position_delta = position_new - position_old + np.array([2, 2])

    # access reverse
    idx = np.ravel_multi_index(position_delta, [5, 5])
    # score_last_step = np.repeat(score_last_step[:,:,np.newaxis], NEIGHBOR, axis=2)
    score_last_step = score_last_step.unsqueeze(2).repeat(1, 1, NEIGHBOR)
    score_all = score_precalculate[idx] + score_last_step
    # print("viterbi reverse")
    score_viterbi = viterbi_reverse(score_all)  # shape (HEIGHT, WIDTH, NEIGHBOR)
    # print("find max")
    score_this_step, score_traceback = torch.max(
        score_viterbi, dim=2
    )  # shape (HEIGHT, WIDTH), value is float score
    # score_traceback = torch.argmax(score_viterbi, dim=2)  # shape (HEIGHT, WIDTH), value from 0 to 24
    return score_this_step, score_traceback

x = [0,5]
y = [5,0]
ang = getAngle(x,y)
print(ang)
new = rotate(x,ang)
print(new)