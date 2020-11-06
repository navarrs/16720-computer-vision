'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
from submission import (
    eightpoint,
    epipolarCorrespondence, 
    essentialMatrix,
    triangulate
)
from helper import (
    displayEpipolarF,
    epipolarMatchGUI,
    camera2
)
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os 

q2_1_file = "q2_1.npz"
q3_1_file = "q3_1.npz"
q3_3_file = "q3_3.npz"
q4_2_file = "q4_2.npz"

coords_file = "../data/templeCoords.npz"
K_file = "../data/intrinsics.npz"
I1 = cv2.imread("../data/im1.png")
I2 = cv2.imread("../data/im2.png")

# Get the fundamental matrix
with np.load(q2_1_file) as data:
    F = data['F']
    print(f"F:\n{F}")

# Get correspondences
with np.load(coords_file) as data:
    x1 = data['x1']
    y1 = data['y1']
    print(f"Reading {len(x1)} points")
    
N = len(x1)
x2 = np.zeros(x1.shape[0])
y2 = np.zeros(y1.shape[0])
for i in range(N):
    x2[i], y2[i] = epipolarCorrespondence(im1=I1, im2=I2, F=F, 
                                          x1=x1[i][0], y1=y1[i][0])

# pts
pts1 = np.zeros((N, 2))
pts1[:, 0] = x1[:, 0]
pts1[:, 1] = y1[:, 0]

pts2 = np.zeros((N, 2))
pts2[:, 0] = x2
pts2[:, 1] = y2

# for i in range(N):
#     print(f"p1({x1[i]}, {y1[i]}), p2({x2[i]}, {y2[i]})")

# Get Cameras
with np.load(K_file) as data:
    K1 = data['K1']
    K2 = data['K2']
    M1 = np.array([[1.0, 0.0, 0.0, 0.0], 
                   [0.0, 1.0, 0.0, 0.0], 
                   [0.0, 0.0, 1.0, 0.0]]) 
    C1 = K1 @ M1
    print(f"C1:\n{C1}")
    
# Triangulate 
with np.load(q3_1_file) as data:
    E = data['E']
    print(f"E:\n{E}")
    
M2s = camera2(E)
for i in range(4):
    M2 = M2s[:, :, i]
    C2 = K2 @ M2
    X, err = triangulate(C1=C1, pts1=pts1, C2=C2, pts2=pts2)
    
    # Valid solution would be the one where z is positive. Thus, the points 
    # are in front of both cameras. 
    if np.all(X[:, -1] > 0):
        print(f"M2 found for i={i+1}")
        print(f"C1:\n{C1}\nC2:\n{C2}\nM2:\n{M2}")
        break

np.savez(q4_2_file, F=F, M1=M1, C1=C1, M2=M2, C2=C2)

# sanity check
with np.load(q4_2_file) as data:
    F_ = data['F']
    C1_ = data['C1']
    M1_ = data['M1']
    C2_ = data['C2']
    M2_ = data['M2']
    assert np.isclose(np.sum(abs(F-F_)), 0.0), "Not matched F"
    assert np.isclose(np.sum(abs(C1-C1_)), 0.0), "Not matched C1"
    assert np.isclose(np.sum(abs(M1-M1_)), 0.0), "Not matched M1"
    assert np.isclose(np.sum(abs(C2-C2_)), 0.0), "Not matched C2"
    assert np.isclose(np.sum(abs(M2-M2_)), 0.0), "Not matched M2"

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Point cloud reconstruction')
ax.scatter(X[:, 0].tolist(), X[:, 1].tolist(), X[:, 2].tolist(), s=3)
plt.show()
plt.close()