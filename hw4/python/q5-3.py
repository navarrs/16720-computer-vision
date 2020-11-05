'''
Q5.3
'''
from submission import (
    ransacF,
    essentialMatrix,
    triangulate,
    bundleAdjustment,
)
from helper import (
    displayEpipolarF,
    camera2,
)
import numpy as np
import cv2
import os 

OUTDIR = "../out/"
DATADIR = "../data/"
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
print(f"F:\n{F}")

# Get Cameras
with np.load("../data/intrinsics.npz") as data:
    K1 = data['K1']
    K2 = data['K2']
    M1 = np.array([[1.0, 0.0, 0.0, 0.0], 
                   [0.0, 1.0, 0.0, 0.0], 
                   [0.0, 0.0, 1.0, 0.0]]) 
    C1 = K1 @ M1
    print(f"K1:\n{K1}\nK2:\n{K2}")
    
E = essentialMatrix(F, K1, K2)
print(f"E:\n{E}")

M2s = camera2(E)
for i in range(4):
    M2_init = M2s[:, :, i]
    C2 = K2 @ M2_init
    P_init, err = triangulate(C1=C1, pts1=pts1, C2=C2, pts2=pts2)
    
    # Valid solution would be the one where z is positive. Thus, the points 
    # are in front of both cameras. 
    if np.all(P_init[:, -1] > 0):
        print(f"M2_init found for i={i+1}:\n{M2_init}")
        break

M2, P_opt = bundleAdjustment(K1=K1, M1=M1, p1=pts1, K2=K2, 
                             M2_init=M2_init, p2=pts2, P_init=P_init)

# Plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_title('Point cloud reconstruction')
# ax.scatter(P_init[:, 0].tolist(), P_init[:, 1].tolist(), P_init[:, 2].tolist(), 'b')
# ax.scatter(P_opt[:, 0].tolist(), P_opt[:, 1].tolist(), P_opt[:, 2].tolist(), 'r')
# plt.show()
# plt.close()