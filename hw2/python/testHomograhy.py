import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from planarH import *
from opts import get_opts

opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
# print(locs1.shape, locs2.shape)
# print(matches)
# display matched features
# plotMatches(cv_cover, cv_desk, matches, locs1, locs2, opts)

# Homography
# # H2to1 = computeH(x1, x2)
# # H2to1 = computeH_norm(x1, x2)
locs1 = locs1[matches[:, 0]]
locs2 = locs2[matches[:, 1]]
#bestH2to1, inliers = 
computeH_ransac(locs1, locs2, opts)

