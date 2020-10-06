import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
import os
#Import necessary functions
from matchPics import matchPics
from helper import plotMatches
from planarH import *


def ablation(opts, cv_cover, cv_desk, hp_cover, locs1, locs2):
  iters = [10, 100, 200, 500, 1000]
  tols  = [1, 2, 5, 10]
  count = 0
  n_tests = len(iters)*len(tols)
  for i in iters:
    for t in tols:
      count += 1
      print(f"Test {count}/{n_tests} with tol: {t}, max_iters: {i}") 
      opts.max_iters = i
      opts.inlier_tol = t
      
      H, inliers = computeH_ransac(locs1, locs2, opts)
      n_inliers = np.sum(inliers)
      print(f"Best H:\n{H}\n Inliers: {n_inliers}")
      composite_img = compositeH(H, hp_cover, cv_desk)
      name = f"hp_tol-{t}_iters-{i}_ninlier-{n_inliers}.png"
      out_file = os.path.join(opts.outdir, name)
      cv2.imwrite(out_file, composite_img)
      

#Write script for Q2.2.4
opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')
hp_cover = cv2.resize(hp_cover, dsize=(cv_cover.shape[1], cv_cover.shape[0]))

matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
# plotMatches(cv_cover, hp_cover, matches, locs1, locs2, opts)
locs1 = locs1[matches[:, 0]]
locs2 = locs2[matches[:, 1]]
locs1[:, [0, 1]] = locs1[:, [1,0]]
locs2[:, [0, 1]] = locs2[:, [1,0]]

if opts.ablation:
  ablation(opts, cv_cover, cv_desk, hp_cover, locs1, locs2)
else:  
  H, inliers = computeH_ransac(locs1, locs2, opts)
  print(f"Best H:\n{H}\n Inliers: {np.sum(inliers)}")

  # H = computeH(locs2, locs)
  # H_2  = computeH_norm(locs1, locs2)
  # Hcv, s = cv2.findHomography(locs1, locs2)
  # print(f"H:\n{H}\nHn:{H_2}\nHcv:{Hcv}")
  
  # Warp image
  warped = cv2.warpPerspective(hp_cover, np.linalg.inv(H), 
                              (cv_desk.shape[1], cv_desk.shape[0]))
  cv2.imshow("warped", warped)
  cv2.waitKey(0)
  
  # Composite
  composite_img = compositeH(H, hp_cover, cv_desk)
  cv2.imshow("composite_img", composite_img)
  cv2.waitKey(0)