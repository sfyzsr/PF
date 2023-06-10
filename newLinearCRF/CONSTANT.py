SAVE_DIR = "newLinearCRF/output/viz_crf_nomap_20"
import os
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print("Directory created:", SAVE_DIR)

WIDTH = 238
HEIGHT = 168
NEIGHBOR_SHIFT = 2  # 2
NEIGHBOR_LENGTH = NEIGHBOR_SHIFT * 2 + 1  # 5
NEIGHBOR = NEIGHBOR_LENGTH * NEIGHBOR_LENGTH  # 25
SEQ_LENGTH = 10000

#######################
# Hyper Parameters
#######################
WEIGHT_TRANSITION = -9.0
WEIGHT_UNARY = -0.3
WEIGHT_HEADING = 0.8
WEIGHT_DISTANCE = -0.6
WEIGHT_LOC = 20.0
BIAS_DISTANCE = 1.0
HEADING_FILLNA = 0.0  # what value to fill NaN values in heading scores