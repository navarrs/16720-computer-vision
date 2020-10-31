'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
from submission import (
    eightpoint
)
from helper import (
    displayEpipolarF
)
import numpy as np
import cv2
import os 

OUTDIR = "../out/q2/"
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

im1 = cv2.imread("../data/im1.png")
im2 = cv2.imread("../data/im2.png")

M = max(im1.shape[0], im1.shape[1])

with np.load("../data/some_corresp.npz") as data:
    pts1 = data['pts1']
    pts2 = data['pts2']
    
F = eightpoint(pts1=pts1, pts2=pts2, M=M)
np.savez(os.path.join(OUTDIR, "F.npz"), F)

im1 = im1[::, ::, ::-1]
im2 = im2[::, ::, ::-1]
displayEpipolarF(im1, im2, F)