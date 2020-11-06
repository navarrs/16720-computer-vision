'''
Tesing homework questions
'''
from submission import (
    eightpoint,
    essentialMatrix,
    triangulate,
    ransacF,
    rodrigues, 
    invRodrigues,
    bundleAdjustment
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
q4_1_file = "q4_1.npz"


def q2_1(I1, I2, M):

    # Find the fundamental matrix
    with np.load("../data/some_corresp.npz") as data:
        pts1 = data['pts1']
        pts2 = data['pts2']

    F = eightpoint(pts1=pts1, pts2=pts2, M=M)
    np.savez(q2_1_file, F=F, M=M)

    # sanity check
    with np.load(q2_1_file) as data:
        F = data['F']
        assert F.shape == (3, 3), f"F shape is {F.shape} instead of (3, 3)"
        M = data['M']
        print(f"F:\n{F}\nM:{M}")

    # Display epipolar lines
    I1 = I1[::, ::, ::-1]
    I2 = I2[::, ::, ::-1]
    displayEpipolarF(I1, I2, F)


def q3_1():

    with np.load(q2_1_file) as data:
        F = data['F']
        print(f"F:\n{F}")

    with np.load("../data/intrinsics.npz") as data:
        K1 = data['K1']
        K2 = data['K2']
        print(f"K1:\n{K1}\nK2:\n{K2}")

    E = essentialMatrix(F=F, K1=K1, K2=K2)
    np.savez(q3_1_file, E=E)

    # sanity check
    with np.load(q3_1_file) as data:
        E = data['E']
        assert E.shape == (3, 3), f"E shape is {E.shape} instead of (3, 3)"
        print(f"E:\n{E}")


def q4_1(I1, I2):
    
    if os.path.exists(q4_1_file):
      with np.load(q4_1_file) as data:
        F = data['F']
        assert F.shape == (3, 3), f"F shape is {F.shape} instead of (3, 3)"
        
        pts1 = data['pts1']
        pts2 = data['pts2']
        assert pts1.shape == pts2.shape, \
            f"pt1s shape {pts1.shape} != pts2 size {pts2.shape}"
            
        print(f"F:\n{F} matched {len(pts1)} points")
    else:
      
      if os.path.exists(q2_1_file):
          with np.load(q2_1_file, allow_pickle=True) as data:
              F = data['F']
      else:
          with np.load("../data/some_corresp.npz") as data:
              pts1 = data['pts1']
              pts2 = data['pts2']

              M = max(I1.shape[0], I1.shape[1])
              F = eightpoint(pts1=pts1, pts2=pts2, M=M)
              np.savez(q2_1_file, F=F, M=M)

          # Epipolar matching
          I1 = I1[::, ::, ::-1]
          I2 = I2[::, ::, ::-1]
          epipolarMatchGUI(I1=I1, I2=I2, F=F)

def q5_1(I1, I2, M):
    with np.load("../data/some_corresp_noisy.npz") as data:
        pts1 = data['pts1']
        pts2 = data['pts2']

    inliers = len(pts1)
    # F = eightpoint(pts1, pts2, M)
    
    F, inliers = ransacF(pts1=pts1, pts2=pts2, M=M, nIters=200, tol=4.5)
    print(f"F:\n{F} matched {np.sum(inliers)}/{len(pts1)} points")

    I1 = I1[::, ::, ::-1]
    I2 = I2[::, ::, ::-1]
    displayEpipolarF(I1, I2, F)
    
def q5_2():
    # test rodrigues
    import random
    rx = np.radians(random.randint(0, 180))
    ry = np.radians(random.randint(0, 180))
    rz = np.radians(random.randint(0, 180))
    
    r = np.array([rx, ry, rz]).T
    R = rodrigues(r)
    r_ = invRodrigues(R)
    
    assert np.allclose(r, r_), f"r_out: {r_} != r_in: {r}"
    
    print(f"r: {r}\nR: {R}\nr_: {r_}")

def q5_3(I1, I2, M):
    with np.load("../data/some_corresp_noisy.npz") as data:
        pts1 = data['pts1']
        pts2 = data['pts2']
    
    F, inliers = ransacF(pts1=pts1, pts2=pts2, M=M, nIters=200, tol=4.5)
    print(f"F:\n{F} matched {np.sum(inliers)}/{len(pts1)} points")
    
    # Keep inliers only
    pts1 = pts1[inliers]
    pts2 = pts2[inliers]
    
    # Get Cameras
    with np.load("../data/intrinsics.npz") as data:
        K1 = data['K1']
        K2 = data['K2']
        print(f"K1:\n{K1}\nK2:\n{K2}")
        
    M1 = np.array([[1.0, 0.0, 0.0, 0.0], 
                    [0.0, 1.0, 0.0, 0.0], 
                    [0.0, 0.0, 1.0, 0.0]]) 
    C1 = K1 @ M1
    
    E = essentialMatrix(F, K1, K2)
    print(f"E:\n{E}")

    M2s = camera2(E)
    for i in range(4):
        M2_init = M2s[:, :, i]
        C2 = K2 @ M2_init
        P_init, err_init = triangulate(C1=C1, pts1=pts1, C2=C2, pts2=pts2)
    
        # Valid solution would be the one where z is positive. Thus, the points 
        # are in front of both cameras. 
        if np.all(P_init[:, -1] > 0):
            print(f"M2_init found for i={i+1}:\n{M2_init}\nError:{err_init}")
            break
    
    
    M2, P_opt = bundleAdjustment(K1=K1, M1=M1, p1=pts1, K2=K2, 
                                 M2_init=M2_init, p2=pts2, P_init=P_init)
    C2 = K2 @ M2
    P_opt_, err_opt = triangulate(C1=C1, pts1=pts1, C2=C2, pts2=pts2)
    print(f"M2_opt:\n{M2}\nError:{err_opt}\nP_opt: {P_opt.shape}\n")

    # Plot
    fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='3d'))
    ax[0].set_title(f'Initial - Rep. err: {round(err_init, 5)} ')
    ax[1].set_title(f'Optimized - Rep. err: {round(err_opt, 5)} ')
    ax[0].scatter(P_init[:, 0].tolist(), P_init[:, 1].tolist(), P_init[:, 2].tolist(), 'b')
    ax[1].scatter(P_opt[:, 0].tolist(), P_opt[:, 1].tolist(), P_opt[:, 2].tolist(), 'r')
    plt.show()
    plt.close()
    

if __name__ == "__main__":

    I1 = cv2.imread("../data/im1.png")
    I2 = cv2.imread("../data/im2.png")
    M = max(I1.shape[0], I1.shape[1])

    # required
    # q2_1(I1, I2, M)
    # q3_1()
    # q4_1(I1, I2)
    
    # extra
    # q5_1(I1, I2, M)
    # q5_2()
    # q5_3(I1, I2, M)
    
    
