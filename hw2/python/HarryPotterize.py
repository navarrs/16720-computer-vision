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

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
# plotMatches(cv_cover, hp_cover, matches, locs1, locs2, opts)
locs1 = locs1[matches[:, 0]]
locs2 = locs2[matches[:, 1]]
locs1[:, [0, 1]] = locs1[:, [1,0]]
locs2[:, [0, 1]] = locs2[:, [1,0]]

# H = computeH(locs2, locs)
# H_2  = computeH_norm(locs1, locs2)
# Hcv, s = cv2.findHomography(locs1, locs2)
# print(f"H:\n{H}\nHn:{H_2}\nHcv:{Hcv}")

H, inliers = computeH_ransac(locs1, locs2, opts)
print(f"Best H:\n{H}\n Inliers: {np.sum(inliers)}")

warped = cv2.warpPerspective(hp_cover, np.linalg.inv(H), 
                             (cv_desk.shape[0], cv_desk.shape[1]))
cv2.imshow("warped", warped)
cv2.waitKey(0)

composite_img = compositeH(H, hp_cover, cv_desk)
cv2.imshow("composite_img", composite_img)
cv2.waitKey(0)
