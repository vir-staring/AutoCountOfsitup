import cv2
import numpy as np
import random
import copy
import math
from scipy.stats import mode
from math import fabs, sin, cos, radians


class NormalizeImage(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean).astype(np.float32)
        self.std = np.array(std).astype(np.float32)
        pass

    def __call__(self, image, labels=None):
        image = image - self.mean
        image = image / self.std
        return image, labels