'''
Q5.1:
'''
from submission import (
    ransacF,
)
from helper import (
    displayEpipolarF,
)
import numpy as np
import cv2
import os 

OUTDIR = "../out/"
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

I1 = cv2.imread("../data/im1.png")
I2 = cv2.imread("../data/im2.png")

# Find the fundamental matrix
M = max(I1.shape[0], I1.shape[1])
with np.load("../data/some_corresp_noisy.npz") as data:
    pts1 = data['pts1']
    pts2 = data['pts2']


F, intiers = ransacF(pts1=pts1, pts2=pts2, M=M, nIters=200, tol=0.001)
print(F)

I1 = I1[::, ::, ::-1]
I2 = I2[::, ::, ::-1]
displayEpipolarF(I1, I2, F)