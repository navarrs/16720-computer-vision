"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
from util import refineF
import random
import scipy

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
        A1 = P1 @ C1
        
        p2 = pts2h[i]
        P2 = np.array([[0, -p2[2], p2[1]], 
                       [p2[2], 0, -p2[0]], 
                       [-p2[1], p2[0], 0]], dtype=np.float)
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
    err1 = (pts1h - pts1_rep)**2
    # print(pts1h, pts1_rep)
    
    err2 = (pts2h - pts2_rep)**2
    err = np.sum(err1) + np.sum(err2)
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
    def gauss(l=3, sig=1., channels=3):
        G = np.zeros((l, l, channels))
        k = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        X, Y = np.meshgrid(k, k)
        kernel = np.exp(-0.5 * (np.square(X) + np.square(Y)) / np.square(sig))
        kernel = kernel / np.sum(kernel)
        G[:, :, 0] = kernel
        G[:, :, 1] = kernel
        G[:, :, 2] = kernel
        return G
    
    H, W, _ = im1.shape
    w = 15
    N = 40
    
    # Homogeneous coord
    p = np.array([x1, y1, 1]).T
    
    # line = ax + by + c
    line = F @ p
    n = np.sqrt(line[0]**2 + line[1]**2)
    line /= n
    if np.isclose(n, 0):
        print("Invalid line")
        return 
    
    y_ = np.arange(y1-N, y1+N)
    x_ = -(line[1] * y_  + line[2]) / line[0]
    # print(x_)
    
    # Search on line
    kernel = gauss(2*w)
    im1_patch = im1[y1-w:y1+w, x1-w:x1+w]# np.multiply(kernel, im1[y1-w:y1+w, x1-w:x1+w])
    x2, y2 = 0, 0
    diff = np.inf
    for i in range(len(y_)):
        x, y = int(x_[i]), int(y_[i])
        if (x - w < 0) or (x + w > W):
            continue
        im2_patch = im2[y-w:y+w, x-w:x+w]#np.multiply(kernel, im2[y-w:y+w, x-w:x+w])
        
        d = np.sqrt(np.sum((im2_patch - im1_patch)**2))
        if diff > d:
            x2, y2 = x, y
            diff = d 
    return x2, y2

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    N = len(pts1)
    best_inliers = 0
    max_inliers = 0
    
    pts1h = np.c_[pts1, np.ones(N)]
    pts2h = np.c_[pts2, np.ones(N)]
    for i in range(nIters):
        # Get samples
        samples = random.sample(range(N), 8)
        p1 = pts1[samples]
        p2 = pts2[samples]
        
        # Inlier counter 
        inliers = np.zeros(N, dtype=np.bool)
        inliers[samples] = True
        
        # Compute the fundamental matrix
        F = eightpoint(p1, p2, M)
        
        # Inliers
        for j in range(N):
            p1 = pts1h[j]
            p2 = pts2h[j]
            if abs(p2 @ F @ p1.T) < tol:
                inliers[j] = True
        
        inlier_count = np.sum(inliers)
        if inlier_count > max_inliers:
            max_inliers = inlier_count
            best_inliers = inliers
    
    p1 = pts1[best_inliers == True]
    p2 = pts2[best_inliers == True]
    F  = eightpoint(p1, p2, M)
    return F, best_inliers.reshape(N, 1)
        
'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta = np.linalg.norm(r)
    I = np.identity(3, dtype=np.float)
    if np.isclose(theta, 0.0):
        return I
    u = (r / theta).reshape(3, 1)
    ux = np.array([[0., -u[2], u[1]],
                   [u[2], 0., -u[0]],
                   [-u[1], u[0], 0.]], dtype=np.float)
  
    c = np.cos(theta)
    s = np.sin(theta)
    # print(c, s)
    # print(u @ u.T)
    return c * I + (1 - c) * (u @ u.T) + s * ux 

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    def S(r):
        if (np.linalg.norm(r) 
            and (r[0] == r[1] == 0.0 and r[2] < 0)
            or  (r[0] == 0.0 and r[1] < 0.0)
            or  (r[0] < 0.0)
        ):
            return -r
        return r
    
    if not np.allclose(abs(R @ R.T - np.identity(3)), 0.0):
        print("Invalid rotation matrix")
        return 
    
    A = (R - R.T) / 2.0
    a = np.array([A[2, 1], A[0, 2], A[1, 0]]).T 
    
    s = np.linalg.norm(a)
    c = (np.sum(np.diag(R)) - 1.0) / 2.0
    # print(s, c)
    
    if np.isclose(c, 1.0) and np.isclose(s, 0.0):
        return np.zeros((3, 1), dtype=np.float)
    
    if np.isclose(c, -1.0) and np.isclose(s, 0.0):    
        for i in range(3):
            v = (R + np.identity(3))[:, i]
            if not np.allclose(v, 0.0):
                break
        u = v / np.linalg.norm(v)
        return S(np.pi * u)
    
    u = a / s
    theta = np.arctan2(s, c)
    return theta * u      
    
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
    N = len(p1)
    C1 = K1 @ M1
    
    r = x[N*3:N*3 + 3]
    t = x[-3].reshape((3, 1))
    M2 = np.hstack([rodrigues(r), t])
    C2 = K2 @ M2
    
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
    P_init = np.vstack([P_init, invRodrigues(M2_init[:3, :3])])
    P_init = np.vstack([P_init, M2_init[:3, -1]])
    f = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x)**2).sum()
    res = scipy.optimize.leastsq(f, P_init)
    M2 = np.zeros((3, 4))
    M2[:3, :3] = rodrigues(res[-2:, :])
    M2[:3,  3] = res[-1, :]
    return M2, res[:-2, :]
