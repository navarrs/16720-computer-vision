import numpy as np
import cv2

# Import necessary functions
from opts import get_opts
import os

from loadVid import loadVid
from matchPics import matchPics
from helper import plotMatches
from planarH import *

# Write script for Q3.1
opts = get_opts()
cv_cover = cv2.imread('../data/cv_cover.jpg')

print("Loading book video...")
bk_frames = loadVid('../data/book.mov')
print("Done.\nLoading AR source video...")
ar_frames = loadVid('../data/ar_source.mov')
print("Done.")

frameSize = (640, 480)
video_out = cv2.VideoWriter('../out/q3-1/ar_out_2.avi',
                            cv2.VideoWriter_fourcc(*'XVID'), 24, frameSize)

n_frames = len(bk_frames) if len(bk_frames) < len(ar_frames) else len(ar_frames)
h1, w1, _ = cv_cover.shape
h2, w2, _ = ar_frames[0].shape

print("Writing AR video...")
for i in range(n_frames):
    print(f"Frame {i}/{n_frames}")
    
    ar_frame = ar_frames[i]
    ar_frame = ar_frame[50:-50, int((w2-w1)/2):int((w2+w1)/2)]
    ar_frame = cv2.resize(ar_frame, dsize=(w1, h1))
    # print(ar_cover.shape, h1)

    # cv2.imshow('ar_c',ar_cover)

    # Match features
    matches, locs1, locs2 = matchPics(cv_cover, bk_frames[i], opts)
    # plotMatches(cv_cover, hp_cover, matches, locs1, locs2, opts)
    locs1 = locs1[matches[:, 0]]
    locs1[:, [0, 1]] = locs1[:, [1, 0]]
    locs2 = locs2[matches[:, 1]]
    locs2[:, [0, 1]] = locs2[:, [1, 0]]

    # Homography and warping
    H, inliers = computeH_ransac(locs1, locs2, opts)
    composite_img = compositeH(H, ar_frame, bk_frames[i])
    # cv2.imshow('warped', composite_img)
    video_out.write(composite_img)

video_out.release()
print("Done...")