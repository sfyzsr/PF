import math
import numpy as np
from tqdm import tqdm
import os
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import time

import CONSTANT
import CRF_utils

DEVICE = torch.device("cuda")
CPU = torch.device("cpu")

DEVICE=CPU

#######################
# Code Parameters
#######################
SAVE_INTERVAL = 1  # save every 50 steps
SAVE_DIR = CONSTANT.SAVE_DIR
METER2PIX = 39.3701 * 72 / 100

window = 35
# useWindow = False
useWindow = True


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

#######################
# Global Scoring Matrix
#######################
HEADING_MATRIX = np.zeros([25, HEIGHT, WIDTH, NEIGHBOR])
DISTANCE_MATRIX = np.zeros([25, HEIGHT, WIDTH, NEIGHBOR])
UNIFIED_SCORE_MATRIX = np.zeros([25, HEIGHT, WIDTH, NEIGHBOR])

#######################
# VISUALIZATION
#######################
# GRAPH_IMG = np.ones([1800, 1800, 3], dtype=np.uint8) * 255
# skeleton = np.load('skeleton_1m.npy')
# for i in skeleton:
#     GRAPH_IMG = cv.circle(GRAPH_IMG, [i[1], i[0]], 2, [150,150,150], 2)
# MAP_IMG = np.load('map_walls_1m.npy')
# HEAT_IMG = np.zeros([HEIGHT, WIDTH, 3], dtype=np.uint8)



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

def meter2pixel(x,y,fig_resolution,fig_scale):

    # fig_scale 1:100

    # fig_resolution 72 pixel / inch

    pix_x=x/0.0254/fig_scale*fig_resolution

    pix_y=-y/0.0254/fig_scale*fig_resolution

    return pix_x,pix_y


def pixel2meter(pix_x,pix_y,fig_resolution,fig_scale):

    # fig_scale 1:100

    # fig_resolution 72 pixel / inch

    x=float(pix_x)/fig_resolution*0.0254*fig_scale

    y=-float(pix_y)/fig_resolution*0.0254*fig_scale

    return x,y

def init_heading():
    # calculate the heading for each grid (unit vector)
    u_vecs = []  # unit vector for map transitions
    neighborhood = np.arange(NEIGHBOR).reshape([NEIGHBOR_LENGTH, NEIGHBOR_LENGTH])
    # calculate the unit distance vector
    for i in range(NEIGHBOR):
        vector = np.array(np.where(neighborhood == i)).flatten() - np.array(
            [NEIGHBOR_SHIFT, NEIGHBOR_SHIFT]
        )
        u_vector = vector / np.linalg.norm(vector)
        u_vecs.append(u_vector)
    u_steps = []  # unit vector for trajectory steps
    neighborhood = np.arange(25).reshape([5, 5])
    for i in range(25):
        vector = np.array(np.where(neighborhood == i)).flatten() - np.array([2, 2])
        u_vector = vector / np.linalg.norm(vector)
        u_steps.append(u_vector)

    # create the heading grid unit_distance * step advancely
    heading_unique = np.matmul(np.array(u_steps), np.array(u_vecs).transpose())
    heading_unique = np.nan_to_num(heading_unique, nan=HEADING_FILLNA)
    for i in range(25):
        HEADING_MATRIX[i] = np.tile(heading_unique[i], (HEIGHT, WIDTH, 1))


def init_distance():
    # calculate the distance for each grid
    neighborhood = np.arange(NEIGHBOR).reshape([NEIGHBOR_LENGTH, NEIGHBOR_LENGTH])
    distances = []
    for i in range(NEIGHBOR):
        vector = np.array(np.where(neighborhood == i)).flatten() - np.array(
            [NEIGHBOR_SHIFT, NEIGHBOR_SHIFT]
        )
        distances.append(np.linalg.norm(vector))
    distances = np.array(distances)

    base_distances = []
    neighborhood = np.arange(25).reshape([5, 5])
    for i in range(25):
        vector = np.array(np.where(neighborhood == i)).flatten() - np.array([2, 2])
        base_distances.append(np.linalg.norm(vector))
    for i in range(25):
        DISTANCE_MATRIX[i] = np.tile(
            np.abs(distances - base_distances[i]), (HEIGHT, WIDTH, 1)
        )


def init_unified(map_transition, map_unary):
    # init score grids for map
    # map constraint
    map_transition = np.repeat(map_transition[np.newaxis, :, :, :], 25, axis=0)
    # wall constraint
    map_unary = np.repeat(map_unary[np.newaxis, :, :, :], 25, axis=0)
    score_precalculate = (
        WEIGHT_TRANSITION * map_transition # score for map obstacles
        + WEIGHT_UNARY * map_unary # score for out-of-building
        + WEIGHT_HEADING * HEADING_MATRIX # score for heading
        + WEIGHT_DISTANCE * DISTANCE_MATRIX # score for distance
        + BIAS_DISTANCE # constant bias
    )
    return score_precalculate

def init_unified_without_map():
    # init score grids for map
    # map constraint
    # map_transition = np.repeat(map_transition[np.newaxis, :, :, :], 25, axis=0)
    # wall constraint
    # map_unary = np.repeat(map_unary[np.newaxis, :, :, :], 25, axis=0)
    score_precalculate = (
        # WEIGHT_TRANSITION  # score for map obstacles
        # + WEIGHT_UNARY +  # score for out-of-building
        WEIGHT_HEADING * HEADING_MATRIX # score for heading
        + WEIGHT_DISTANCE * DISTANCE_MATRIX # score for distance
        + BIAS_DISTANCE # constant bias
    )
    return score_precalculate


def index_to_coordinate(position, index):
    # x = index % NEIGHBOR_LENGTH
    # y = index // NEIGHBOR_LENGTH
    y, x = np.unravel_index(index.to(CPU), (NEIGHBOR_LENGTH, NEIGHBOR_LENGTH))
    new_position = (position[0] + y - NEIGHBOR_SHIFT, position[1] + x - NEIGHBOR_SHIFT)
    return new_position


def traceback(traceback_matrices, position):
    trace = []
    while traceback_matrices:  # shape (SEQ_LENGTH, HEIGHT, WIDTH)
        trace.append(position)
        matrix = traceback_matrices.pop()
        index = matrix[position]
        # print("retreived index", index)
        position = index_to_coordinate(position, index)
        # print("now position", position)
    return trace

dataDir = "newLinearCRF/data/"
def main():
    # read data
    mega_trajectory = np.load( dataDir +
        "trajectory_discrete.npy"
    )  # shape (SEQ_LENGTH, 2), [y, x], top-left origin 
    # (575, 2)
    map_transition = np.load( dataDir +
        "map_transition.npy"
    )  # shape (HEIGHT, WIDTH, NEIGHBOR), value 0 or 1, 1 for obstacle
    # (168, 238, 25)
    map_unary = np.load( dataDir +
        "map_unary.npy"
    )  # shape (HEIGHT, WIDTH, NEIGHBOR), values 0 or 1, 1 for outside of the floor plan
    localization = np.load( dataDir +
        "localization.npy"
    )  # shape (SEQ_LENGTH, 5, 2), [y, x], top-left origin
    # (574, 5, 2)
    localization_bool = np.load( dataDir +
        "localization_bool.npy"
    )  # shape (SEQ_LENGTH, ), bool
    # (575,)
    localization_bool = np.append(localization_bool, False)
    # score_matrix = np.ones([HEIGHT, WIDTH])  # shape (HEIGHT, WIDTH)
    score_matrix = torch.ones([HEIGHT, WIDTH], device=DEVICE)
    traceback_all_steps = []
    # score_all_steps = []

    # Init
    SEQ_LENGTH = mega_trajectory.shape[0]
    # SEQ_LENGTH = 100
    init_heading()
    init_distance()
    score_precalculate = init_unified(map_transition, map_unary)
    # score_precalculate = init_unified_without_map()
    # print(score_precalculate)
    
    score_precalculate = torch.from_numpy(score_precalculate).to(DEVICE)

    # init visualization
    # GRAPH_IMG = np.ones([1800, 1800, 3], dtype=np.uint8) * 255
    # skeleton = np.load("skeleton_1m_v2.npy")
    # for i in skeleton:
    #     GRAPH_IMG = cv.circle(GRAPH_IMG, [i[1], i[0]], 1, [150, 150, 150], 1)
    MAP_IMG = np.load(dataDir +"map_original.npy")[:, :, 0]
    MAP_IMG = np.repeat(MAP_IMG[:, :, np.newaxis], 3, axis=2)
    gt_seq = np.load(dataDir +"23_gt.npy")

    online = []

    # a flag use to interval the rotate angle
    rotateFlag = 3

    # Forward
    print("Starting Forward Operations")
    for i in tqdm(range(SEQ_LENGTH - 1)):

        if(i>200 and useWindow == True):
            start = np.unravel_index(score_matrix.argmax().to(CPU), score_matrix.shape)
            traceback_matrices = traceback_all_steps.copy()
            trace = traceback(traceback_matrices, start)
            # traceLen = len(trace)
            # print(traceLen)

            vec_s_list = trace[-window:]
            onlineWindow = online[-window:]

            vec_z_list = mega_trajectory[i-window:i]
            # vec_z_list = mega_trajectory[i-window:i+1]

            if not localization_bool[i+1]:
                score_matrix, traceback_matrix = CRF_utils.score2(
                position_old, position_new, score_matrix, score_precalculate,vec_s_list,vec_z_list
            )
            else:
                # seem like when located in the right position get higher marks ?
                score_matrix, traceback_matrix = CRF_utils.score_loc2(
                    position_old, position_new, score_matrix, score_precalculate, localization[i+1],vec_s_list,vec_z_list
                )
            rotateFlag = 0


            # print(vec_s_list)
            # print(onlineWindow)
            # print("hello")
            # quit()
             
            # rotateFlag = rotateFlag + 1
            # position_old = mega_trajectory[i]
            # position_new = mega_trajectory[i + 1]

            # if rotateFlag == 10: 
            # # do the rotate
            #     if not localization_bool[i+1]:
            #         score_matrix, traceback_matrix = CRF_utils.score2(
            #         position_old, position_new, score_matrix, score_precalculate,vec_s_list,vec_z_list
            #     )
            #     else:
            #         # seem like when located in the right position get higher marks ?
            #         score_matrix, traceback_matrix = CRF_utils.score_loc2(
            #             position_old, position_new, score_matrix, score_precalculate, localization[i+1],vec_s_list,vec_z_list
            #         )
            #     rotateFlag = 0
            # else:
            #     if not localization_bool[i+1]:
            #         score_matrix, traceback_matrix = CRF_utils.score(
            #         position_old, position_new, score_matrix, score_precalculate
            #     )
            #     else:
            #         # seem like when located in the right position get higher marks ?
            #         score_matrix, traceback_matrix = CRF_utils.score_loc(
            #             position_old, position_new, score_matrix, score_precalculate, localization[i+1]
            #         )
        else:

            position_old = mega_trajectory[i]
            position_new = mega_trajectory[i + 1]
            if not localization_bool[i+1]:
                score_matrix, traceback_matrix = CRF_utils.score(
                position_old, position_new, score_matrix, score_precalculate
            )
            else:
                # seem like when located in the right position get higher marks ?
                score_matrix, traceback_matrix = CRF_utils.score_loc(
                    position_old, position_new, score_matrix, score_precalculate, localization[i+1]
                )
        traceback_all_steps.append(
            traceback_matrix.to(CPU)
        )  # Traceback matrices are too large to be stored on GPU

        
        # score_all_steps.append(score_matrix)
        # GRAPH_IMG = cv.circle(
        #     GRAPH_IMG, (position_new[1], position_new[0]), 1, [255, 0, 0], 1
        # )
        if i % SAVE_INTERVAL == 0:
            # print("Tracing back from step", i)
            viz_file_name = "viz" + f"{i:06}" + ".png"
            trace_file_name = "trace" + f"{i:06}" + ".npy"
            start = np.unravel_index(score_matrix.argmax().to(CPU), score_matrix.shape)
            traceback_matrices = traceback_all_steps.copy()
            trace = traceback(traceback_matrices, start)
            online.append(trace[0])
            trace = np.array(trace)
            MAP_IMG_COPY = MAP_IMG.copy()
            # for point in trace:
            #     MAP_IMG_COPY = cv.circle(
            #         MAP_IMG_COPY, (point[1]*10, point[0]*10), 3, [255, 0, 0], 3
            #     )
            # score_all_steps = [i.to(torch.device('cpu')).numpy() for i in score_all_steps]
            # np.save(os.path.join(SAVE_DIR, trace_file_name), score_all_steps)
            # score_all_steps = []
            # visualize(MAP_IMG_COPY, os.path.join(SAVE_DIR, viz_file_name))
            try:
                visualize(MAP_IMG_COPY, i, np.array(online), trace, gt_seq, localization, os.path.join(SAVE_DIR, viz_file_name))
            except IndexError:
                print(f"Index {i // 2} is out of bounds for gt_seq. Exiting the loop.")
                break
            # visualize(MAP_IMG_COPY, GRAPH_IMG, os.path.join(SAVE_DIR, viz_file_name))
            torch.cuda.empty_cache()

    # Backward
    print("Starting Backward Operations")
    start = np.unravel_index(score_matrix.argmax().to(CPU), score_matrix.shape)
    traceback_matrices = traceback_all_steps.copy()
    trace = traceback(traceback_matrices, start)

    print("Saving results")
    # trace = [i.to(torch.device('cpu')).numpy() for i in trace]
    trace = np.array(trace)
    online = np.array(online)

    # to evaluate need to compare with gt not online
    L2s = []
    # Calculate mean L2 distance between estimated trace and real trace
    l2 = np.sqrt(np.sum((trace - online)**2, axis=1))
    L2s.append(l2)
    mean_l2 = np.mean(l2)
    # Calculate accuracy of estimated trace
    threshold = mean_l2  # Set a threshold distance for considering a point "accurate"
    accurate_points = np.sum(l2 < threshold)
    total_points = len(l2)
    accuracy = accurate_points / total_points

    current_time = time.time()
    # plot histogram of trace x-values
    # plt.hist(L2s, bins=20)
    # # add axis labels and title
    # plt.xlabel('l2')
    # plt.ylabel('Frequency')
    # plt.title('L2s')

    # # display the plot
    # plt.savefig("output/Histogram of L2_"+str(current_time)+".png", dpi=150)
    # plt.show()

    # Print results
    print("Mean L2 distance:", mean_l2)
    print("Accuracy:", accuracy)

    np.save(SAVE_DIR+"trace.npy", trace)
    np.save(SAVE_DIR+"online.npy", online)

    # plt.plot(trace[:, 0]*10, trace[:, 1]*10, 'b', label='Trace')
    # plt.plot(online[:, 0]*10, online[:, 1]*10, 'r', label='Online')
    # gt_seq[:, 0] /= 2
    # # gt_seq[:, 0] += 1
    # # plt.plot(gt_seq[:, 0]*METER2PIX, gt_seq[:, 1]*METER2PIX, 'k', label='gt')
    # plt.legend()
    # plt.xlabel('X')
    # plt.ylabel('Y') 
    
    # plt.savefig("newLinearCRF/temp/window "+str(window)+".png", dpi=150)
    # plt.show()

    # scores = np.array(score_all_steps)
    # np.save('score.npy', scores)
    # visualize(MAP_IMG_COPY, i, np.array(online), trace, gt_seq[:, 1], localization, "newLinearCRF/temp/window "+str(window)+".png")
            

def visualize(img1, i, online, trace, gt_seq, localization, name):
    plt.rcParams["figure.figsize"] = (10,8)
    f, axarr = plt.subplots(1, 1)
    plt.imshow(img1.astype('uint8'), alpha=0.3)
    plt.plot(gt_seq[:i//2+1, 0]*METER2PIX, -gt_seq[:i//2+1, 1]*METER2PIX, c='deepskyblue', alpha=0.5, label='GT history')
    plt.scatter(gt_seq[i//2, 0]*METER2PIX, -gt_seq[i//2, 1]*METER2PIX, c='mediumblue', s=20, label='GT now')
    plt.plot(trace[:, 1]*10, trace[:, 0]*10, c='magenta', alpha=0.5, label='offline trajectory')
    plt.scatter(trace[0, 1]*10, trace[0, 0]*10, c='firebrick', s=20, label='online now')
    plt.plot(online[2:, 1]*10, online[2:, 0]*10, c='lightcoral', alpha=0.5, label='online trajectory')
    plt.scatter(localization[i+1, :5, 1]*10, localization[i+1, :5, 0]*10, c='violet', s=10, label='wifi pos')
    # axarr[1].imshow(img2[250:1600, 300:1268, :])
    plt.xlim(600, 2250)
    plt.ylim(1650, 0)
    plt.legend(loc='lower left')
    plt.savefig(name, dpi=150)
    plt.clf()
    plt.close()


if __name__ == "__main__":
    main()
