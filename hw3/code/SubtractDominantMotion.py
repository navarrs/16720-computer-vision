import numpy as np
from scipy.ndimage import (
    binary_erosion, 
    binary_dilation, 
    affine_transform
    )
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
    M = np.linalg.inv(M)
    
    # Match It to It1
    # warpAffine needs a 2x3 matrix
    Iw1 = cv2.warpAffine(image1, M[:2,:], dsize=image2.T.shape) 
    # Iw1 = affine_transform(image1, matrix=M[:2,:2], 
    #                        offset=M[:2, 2], output_shape=image2.shape)
    d_ = abs(image2 - Iw1)
    mask[d_ < tolerance] = 0
    
    # TODO: fix -- 
    # mask = binary_erosion(mask, structure=np.array(([0,1,0],[1,1,1],[0,1,0])))
    # mask = binary_dilation(mask, structure=np.array(([0,1,0],[1,1,1],[0,1,0])))
    # mask = binary_dilation(mask, iterations=1)              
    return mask