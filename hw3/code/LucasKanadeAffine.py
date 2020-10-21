import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array] put your implementation here
    """
    M = np.eye(3)
    
    H, W = It.shape[0], It.shape[1]
    x_, y_ = np.arange(W), np.arange(H)
    
    T = RectBivariateSpline(y_, x_, It)
    I = RectBivariateSpline(y_, x_, It1)
    dIt1dy_, dIt1dx_ = np.gradient(It1)
    dIdx = RectBivariateSpline(y_, x_, dIt1dx_)
    dIdy = RectBivariateSpline(y_, x_, dIt1dy_)
    
    # Create the homogeneous coordinates
    xx_, yy_ = np.meshgrid(x_, y_)
    x = xx_.reshape(W*H)
    y = yy_.reshape(W*H)
    C = np.vstack((x, y, np.ones((W*H))))
    
    for i in range(int(num_iters)):
        
        c = M @ C
        # Mask valid points
        mask = (c[0] >= 0) & (c[1] >= 0) & (c[0] < W) & (c[1] <= H)
        x_, y_ = x[mask], y[mask]
        xw_, yw_ = c[0][mask], c[1][mask]
        
        # Iw
        Iw = I.ev(yw_, xw_)
        T_ = T.ev(y_, x_)
        
        # T - I
        b = (T_ - Iw).reshape(-1, 1)
        
        # dIw
        dI = np.vstack((dIdx.ev(yw_, xw_), dIdy.ev(yw_, xw_)))
        # print(T_.shape, Iw.shape, dI.shape)
        
        # Build A
        X = dI * x_
        Y = dI * y_
        A = np.vstack((X[0], Y[0], dI[0], 
                       X[1], Y[1], dI[1])).T

        H = A.T @ A
        dp = np.linalg.pinv(H) @ A.T @ b
        # print(np.linalg.norm(dp))
        
        M[0, 0] += dp[0]
        M[0, 1] += dp[1]
        M[0, 2] += dp[2]
        M[1, 0] += dp[3]
        M[1, 1] += dp[4]
        M[1, 2] += dp[5]
        
        if np.linalg.norm(dp) <= threshold:
            break
        
    return M