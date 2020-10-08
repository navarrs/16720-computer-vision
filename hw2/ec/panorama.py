import sys
sys.path.insert(1, '../python')

import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
from matchPics import matchPics
from helper import plotMatches
from planarH import *

opts = get_opts()

def stitch(img1, img2, black_removal=False):
  # Images dimensions
  h1, w1, _ = img1.shape
  h2, w2, _ = img2.shape
  
  # Match
  matches, locs1, locs2 = matchPics(img1, img2, opts)
  # plotMatches(img1, img2, matches, locs1, locs2, opts)
  locs1 = locs1[matches[:, 0]]
  locs2 = locs2[matches[:, 1]]
  locs1[:, [0, 1]] = locs1[:, [1,0]]
  locs2[:, [0, 1]] = locs2[:, [1,0]]
  
  # Homography
  H, inliers = computeH_ransac(locs1, locs2, opts)
  
  # Determine final size of panorama
  corners = np.array(
    [[0, 0, 1], [w2, 0, 1], [0, h2, 1], [w2, h2, 1]], dtype=np.float)
  corners = np.matmul(H, corners.T)
  corners = np.divide(corners, corners[-1, :]).T
  # print(corners)
  
  # Warp
  if black_removal:
    # Define the width 
    min_w = int(min(corners[1, 0], corners[3, 0]))
    width_right = max(min_w, w1)
    width = width_right

    # Define the height
    max_h = int(np.max(corners[:2, 1]))
    height_top = max(0, max_h)
    min_h = int(np.min(corners[2:, 1]))
    height_low = min(min_h, h1)
    height = height_low - height_top
    
    pano = cv2.warpPerspective(img2, H, (width, height))
    pano[:height, :w1, :] = img1[:height, :]
    pano = pano[height_top:, :]
  else:
    width = w1 + int(0.5*w2)
    height = h1 + int(0.5*h2)
    pano = cv2.warpPerspective(img2, H, (width, height))
    pano[:h1, :w1, :] = img1
  
  return pano


if __name__ == "__main__":  
  panol = cv2.imread('left.png')
  panor = cv2.imread('right.png')
  # cv2.imshow("panol", panol)
  # cv2.imshow("panor", panor)
  # cv2.waitKey(0)
  
  panorama = stitch(panol, panor, True)
  cv2.imshow("panorama", panorama)
  cv2.waitKey(0)
  
  

