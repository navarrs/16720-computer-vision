"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
from util import refineF

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    N = len(pts1)
    
    # Scale the data 
    pts1n = pts1 / M
    pts2n = pts2 / M
    
    # Create A
    A = np.zeros((N, 9))
    for i in range(N):
        pt1 = pts1n[i]
        x, y = pt1[0], pt1[1]
        pt2 = pts2n[i]
        x_, y_ = pt2[0], pt2[1]
        
        A[i, :] = [x_*x, x_*y, x_, y_*x, y_*y, y_, x, y, 1]
    
    # Find fundamental matrix
    ATA = A.T @ A
    _,_,V_T = np.linalg.svd(ATA)
    # Last column of V is last row of V_T
    F_est = V_T[-1, :].reshape((3, 3))
    # print(F_est)
    
    # Enforce rank2 constraint
    U, S, V_T = np.linalg.svd(F_est, full_matrices=True)
    S[-1] = 0.0
    F = U @ np.diag(S) @ V_T
    # print(F)    
    
    # Refine
    F = refineF(F, pts1n, pts2n)
     
    # Unnormalize
    T = np.array([[1./M, 0, 0], 
                  [0, 1./M, 0], 
                  [0,    0, 1]], dtype=np.float)
    F = T.T @ F @ T
    # print(F)
    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    return K2.T @ F @ K1


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    N = len(pts1)
    # Homogeneous coordinates
    pts1h = np.c_[pts1, np.ones(N)]
    pts2h = np.c_[pts2, np.ones(N)]
    
    A = np.zeros((4, 4), dtype=np.float)
    X = np.zeros((N, 4), dtype=np.float)
    pts1_rep = np.zeros((N, 3), dtype=np.float)
    pts2_rep = np.zeros((N, 3), dtype=np.float)
    
    # Compute the 3D points
    for i in range(N):
        p1 = pts1h[i]
        P1 = np.array([[0, -p1[2], p1[1]], 
                       [p1[2], 0, -p1[0]], 
                       [-p1[1], p1[0], 0]], dtype=np.float)
        p2 = pts2h[i]
        P2 = np.array([[0, -p2[2], p2[1]], 
                       [p2[2], 0, -p2[0]], 
                       [-p2[1], p2[0], 0]], dtype=np.float)
        
        
        A1 = P1 @ C1
        A2 = P2 @ C2
        
        A[:2, :] = A1[:2, :]
        A[2:, :] = A2[:2, :]
        
        _,_,V_T = np.linalg.svd(A)
        X[i, :] = V_T[-1, :]
        
        # Reproject to 2D
        pts1_rep[i] = C1 @ X[i]
        pts1_rep[i, :] = pts1_rep[i, :] / pts1_rep[i, -1]
        pts2_rep[i] = C2 @ X[i]
        pts2_rep[i, :] = pts2_rep[i, :] / pts2_rep[i, -1]
        # print(pts1_rep[i], pts2_rep[i])
        
        X[i, :] = X[i, :] / X[i, -1]
        # print(X[i, :])
        
    # Measure reprojection error
    err1 = np.linalg.norm(pts1h - pts1_rep, ord=2)
    err2 = np.linalg.norm(pts2h - pts2_rep, ord=2)
    err = err1 + err2
    # print(err)

    return X[:, :-1], err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    pass

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
