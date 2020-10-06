import numpy as np
import cv2

#Import necessary functions
from opts import get_opts
import os

from matchPics import matchPics
from helper import plotMatches
from planarH import *


#Write script for Q3.1
opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cap_bk = cv2.VideoCapture('../data/book.mov')
cap_ar = cv2.VideoCapture('../data/ar_source.mov')

frameSize = (640, 480)
video_out = cv2.VideoWriter('../out/q3-1/ar_out_2.avi', 
                            cv2.VideoWriter_fourcc(*'XVID'), 24, frameSize)

h1, w1, _ = cv_cover.shape

c = 0
while cap_bk.isOpened() and cap_ar.isOpened():
  c += 1
  ret1, bk_frame = cap_bk.read()
  ret2, ar_frame = cap_ar.read()

  # print(book_frame.shape, ar_frame.shape)
  if ret1 and ret2:
    # cv2.imshow('ar',ar_frame)
    h2, w2, _ = ar_frame.shape
    
    ar_cover = ar_frame[50:-50, int((w2-w1)/2):int((w2+w1)/2)]
    ar_cover = cv2.resize(ar_cover, dsize=(cv_cover.shape[1], cv_cover.shape[0]))
    # print(ar_cover.shape, h1)
    
    # cv2.imshow('ar_c',ar_cover)
    
    # Match features
    matches, locs1, locs2 = matchPics(cv_cover, bk_frame, opts)
    # plotMatches(cv_cover, hp_cover, matches, locs1, locs2, opts)
    locs1 = locs1[matches[:, 0]]
    locs1[:, [0, 1]] = locs1[:, [1,0]]
    locs2 = locs2[matches[:, 1]]
    locs2[:, [0, 1]] = locs2[:, [1,0]]
    
    # 
    H, inliers = computeH_ransac(locs1, locs2, opts)
    
    composite_img = compositeH(H, ar_cover, bk_frame)
    # cv2.imshow('warped', composite_img)
    video_out.write(composite_img)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  else:
    break
  
cap_bk.release()
cap_ar.release()
video_out.release()
cv2.destroyAllWindows()