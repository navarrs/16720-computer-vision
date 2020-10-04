import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

#Import necessary functions
from matchPics import matchPics
from helper import plotMatches
from planarH import *

#Write script for Q2.2.4
opts = get_opts()

panol = cv2.imread('../data/pano_left.jpg')
panor = cv2.imread('../data/pano_right.jpg')

matches, locs1, locs2 = matchPics(panol, panor, opts)
# plotMatches(panol, panor, matches, locs1, locs2, opts)
locs1 = locs1[matches[:, 0]]
locs2 = locs2[matches[:, 1]]

# H, inliers = computeH_ransac(locs1, locs2, opts)
# H, s = cv2.findHomography(locs1, locs2)

# stitched = stitch(panor, panol, H)

# cv2.imshow("pano", stitched)
cv2.waitKey(0)

