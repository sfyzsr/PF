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

