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

# Find the fundamental matrix
with np.load("../data/some_corresp.npz") as data:
    pts1 = data['pts1']
    pts2 = data['pts2']
N = len(pts1)
    
F_file = "q2-1.npz"
if os.path.exists(F_file):
    with np.load(F_file, allow_pickle=True) as data:  
        F = data['F']
else:
    I1 = cv2.imread("../data/im1.png")
    I2 = cv2.imread("../data/im2.png")
    M = max(I1.shape[0], I1.shape[1])
    F = eightpoint(pts1=pts1, pts2=pts2, M=M)
    # np.savez(F_file, F=F, M=M)
print(f"F:\n{F}")

# Find the essential matrix
with np.load("../data/intrinsics.npz") as data:
    K1 = data['K1']
    K2 = data['K2']
    
E_file = "q3-1.npz"
if os.path.exists(E_file):
    with np.load(E_file, allow_pickle=True) as data:  
        E = data['E']
else:
    E = essentialMatrix(F=F, K1=K1, K2=K2)
    np.savez(E_file, E=E)
print(f"E:\n{E}")

# Triangulate
q3_3_file = "q3_3.npz"
M1 = np.array([[1.0, 0.0, 0.0, 0.0], 
               [0.0, 1.0, 0.0, 0.0], 
               [0.0, 0.0, 1.0, 0.0]]) 
M2s = camera2(E)
C1 = K1 @ M1
for i in range(4):
    M2 = M2s[:, :, i]
    C2 = K2 @ M2
    P, err = triangulate(C1=C1, pts1=pts1, C2=C2, pts2=pts2)
    
    # Valid solution would be the one where z is positive. Thus, the points 
    # are in front of both cameras. 
    if np.all(P[:, -1] > 0):
        print(f"M2 found for i={i+1} with err: {err}")
        np.savez(q3_3_file, M2=M2, C2=C2, P=P)
        
        # sanity check 
        with np.load(q3_3_file) as data:
            M2 = data['M2']
            assert M2.shape == (3, 4), f"M2 shape is {M2.shape} instead of (3, 4)"
            C2 = data['C2']
            assert C2.shape == (3, 4), f"C2 shape is {C2.shape} instead of (3, 4)"
            P = data['P']
            assert P.shape == (N, 3), f"P shape is {P.shape} instead of ({N}, 3)"
            print(f"M2:\n{M2}\nC2:\n{C2}\n")
            # print(P)
        break
        
        