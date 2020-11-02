'''
Q4.1:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
from submission import (
    eightpoint,
    essentialMatrix, 
    triangulate
)
from helper import (
    displayEpipolarF,
    epipolarMatchGUI,
    camera2
)
import numpy as np
import cv2
import os 

I1 = cv2.imread("../data/im1.png")
I2 = cv2.imread("../data/im2.png")

OUTDIR = "../out/"
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

# Find the fundamental matrix
F_file = os.path.join(OUTDIR, "F.npz")
if os.path.exists(F_file):
    with np.load(F_file, allow_pickle=True) as data:  
        F = data['F']
else:
    with np.load("../data/some_corresp.npz") as data:
        pts1 = data['pts1']
        pts2 = data['pts2']
    M = max(I1.shape[0], I1.shape[1])
    F = eightpoint(pts1=pts1, pts2=pts2, M=M)
    np.savez(os.path.join(OUTDIR, "F.npz"), F=F)
print(f"F:\n{F}")

# Epipolar matching
I1 = I1[::, ::, ::-1]
I2 = I2[::, ::, ::-1]

epipolarMatchGUI(I1=I1, I2=I2, F=F)