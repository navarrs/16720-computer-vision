import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
import cv2 

from LucasKanadeAffine import LucasKanadeAffine as LKA

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    M = LKA(image1, image2, threshold, num_iters)
    
    # Match It to It1
    # warpAffine needs a 2x3 matrix
    # M = np.linalg.inv(M)
    Iw1 = cv2.warpAffine(image2, M[:2,:], dsize=image1.shape) 
    d_ = abs(image2 - Iw1)
    mask[d_ < tolerance] = 0
    
    mask = binary_dilation(mask, iterations=2)
    mask = binary_erosion(mask, iterations=3)
    return mask