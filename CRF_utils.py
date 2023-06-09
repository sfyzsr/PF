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
    x = vector[0] * math.cos(angle) - vector[1] * math.sin(angle)
    y = vector[0] * math.sin(angle) + vector[1] * math.cos(angle)
    # if(x<0):
    #     x = vx
    #     y = vy
    # if(y<0):
    #     x = vx
    #     y = vy
    # if(x>5):
    #     x = vx
    #     y = vy
    # if(y>5):
    #     x = vx
    #     y = vy
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
  

def correctAngle(S,Z):
    
    # print(Z)
    # print("")
    # print(S)
    Slen = len(S)
    
    sum = 0
    for i in range(Slen-1,1,-1):
        x1 = S[i] [0] - S[i-1] [0]
        y1 = S[i] [1] - S[i-1] [1]

        x2 = Z[i] [0] - Z[i-1] [0]
        y2 = Z[i] [1] - Z[i-1] [1]
        a = getAngle([x1,y1],[x2,y2])
        
        sum += a
    
    avg = sum/Slen
    # print(avg)
    return avg
        

def score_loc2(position_old, position_new, score_last_step, score_precalculate, localizations,vec_s_list,vec_z_list):
    
    position_diff = position_new - position_old 
    angle = correctAngle(vec_s_list,vec_z_list)
    position_rotate = rotate(position_diff,angle)

    position_delta = position_rotate + np.array([2, 2]) 

    # position_delta = position_new - position_old + np.array([2, 2])

    # print("aaa")
    # print(position_delta)
    # angle = correctAngle(vec_s_list,onlineWindow)
    # position_delta = rotate(position_delta,angle)
    # print(vec_s_list)
    # print(vec_z_list)
    # print(angle)
    # print(position_delta)

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

    # position_delta = position_new - position_old + np.array([2, 2]) # here change the position vector to the grid
    # print(position_diff)
    # print(position_delta)
    # print("score2")
    # print(position_delta)

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


# def calculate_score_heading(position_old, position_new):
#     # only has 8 posibilities
#     position_delta = position_new - position_old + np.array([1, 1])
#     idx = np.ravel_multi_index(position_delta, [3, 3])
#     score_heading = WEIGHT_HEADING * HEADING_MATRIX[idx]
#     return score_heading


# def calculate_score_distance(position_old, position_new):
#     # pre_calculated
#     position_delta = position_new - position_old + np.array([1, 1])
#     idx = np.ravel_multi_index(position_delta, [3, 3])
#     score_distance = WEIGHT_DISTANCE * DISTANCE_MATRIX[idx] + BIAS_DISTANCE
#     return score_distance


# def calculate_score_obstacle(map_transition):
#     # same as or linear to map_transition
#     score_obstacle = WEIGHT_TRANSITION * map_transition
#     return score_obstacle


# def calculate_score_boundary(map_unary):
#     # same as or linear to map_unary
#     score_boundary = WEIGHT_UNARY * map_unary
#     return score_boundary


# def calculate_score_overrall(score_heading, score_distance, score_obstacle, score_boundary, score_last_step):
#     # log of multiplied exp score, resulting in simple summation
#     score_last_step = np.repeat(score_last_step[:,:,np.newaxis], NEIGHBOR, axis=2)
#     score_all = score_heading + score_distance + score_obstacle + score_boundary + score_last_step
#     return score_all