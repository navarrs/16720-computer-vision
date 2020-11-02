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

OUTDIR = "../out"
DATADIR = "../data"
F_file = os.path.join(OUTDIR, "F.npz")
E_file = os.path.join(OUTDIR, "E.npz")
Q33_file = os.path.join(OUTDIR, "q3_3.npz")

coords_file = os.path.join(DATADIR, "templeCoords.npz")
K_file = os.path.join(DATADIR, "intrinsics.npz") 

I1 = cv2.imread("../data/im1.png")
I2 = cv2.imread("../data/im2.png")

# Get the fundamental matrix
with np.load(F_file) as data:
    F = data['F']
    print(f"F:\n{F}")

# Get correspondences
with np.load(coords_file) as data:
    x1 = data['x1']
    y1 = data['y1']
    
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
E = essentialMatrix(F, K1, K2)
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
    
# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Point cloud reconstruction')
ax.scatter(X[:, 0].tolist(), X[:, 1].tolist(), X[:, 2].tolist())
plt.show()
plt.close()