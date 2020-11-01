'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import cv2
import numpy as np
import os 

from helper import camera2
from submission import (
    eightpoint,
    essentialMatrix, 
    triangulate
)

OUTDIR = "../out/"
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

# Find the fundamental matrix
with np.load("../data/some_corresp.npz") as data:
    pts1 = data['pts1']
    pts2 = data['pts2']
    
F_file = os.path.join(OUTDIR, "F.npz")
if os.path.exists(F_file):
    with np.load(F_file, allow_pickle=True) as data:  
        F = data['F']
else:
    I1 = cv2.imread("../data/im1.png")
    I2 = cv2.imread("../data/im2.png")
    M = max(I1.shape[0], I1.shape[1])
    F = eightpoint(pts1=pts1, pts2=pts2, M=M)
    np.savez(os.path.join(OUTDIR, "F.npz"), F=F)
print(f"F:\n{F}")

# Find the essential matrix
with np.load("../data/intrinsics.npz") as data:
    K1 = data['K1']
    K2 = data['K2']
    
E_file = os.path.join(OUTDIR, "E.npz")
if os.path.exists(F_file):
    with np.load(E_file, allow_pickle=True) as data:  
        E = data['E']
else:
    E = essentialMatrix(F=F, K1=K1, K2=K2)
    np.savez(os.path.join(OUTDIR, "E.npz"), F)
    
print(f"E:\n{E}")

# Triangulate
q3_3_file = os.path.join(OUTDIR, "q3_3.npz")
m = 4
M1 = np.array([[1.0, 0.0, 0.0, 0.0], 
               [0.0, 1.0, 0.0, 0.0], 
               [0.0, 0.0, 1.0, 0.0]]) 
M2s = camera2(E)
C1 = K1 @ M1
for i in range(m):
    M2 = M2s[:, :, i]
    C2 = K2 @ M2
    P, err = triangulate(C1=C1, pts1=pts1, C2=C2, pts2=pts2)
    
    # Valid solution would be the one where z is positive. Thus, the points 
    # are in front of both cameras. 
    if np.all(P[:, -1] > 0):
        print(f"M2 found for i={i+1}")
        np.savez(q3_3_file, M2=M2, C2=C2, P=P)
        print(f"C1:\n{C1}\nC2:\n{C2}\nM2:\n{M2}")
        break
        
        