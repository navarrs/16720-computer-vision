import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array]
    """

    # put your implementation here
    M = np.eye(3)
    
    # Pre-compute actions
    H, W = It.shape[0], It.shape[1]
    x_, y_ = np.arange(W), np.arange(H)
    
    T = RectBivariateSpline(y_, x_, It)
    I = RectBivariateSpline(y_, x_, It1)
    
    xx_, yy_ = np.meshgrid(x_, y_)
    x = xx_.reshape(W*H)
    y = yy_.reshape(W*H)
    C = np.vstack((x, y, np.ones((W*H))))
    T_ = T.ev(y, y)
    
    # dTdx = T.ev(y, x, dy=1)
    # dTdy = T.ev(y, x, dx=1)
    # print(dTdx.shape, dTdy.shape)
    dT = np.vstack((T.ev(y, x, dy=1), T.ev(y, x, dx=1)))
    # print(dT.shape)
    X = dT * x
    Y = dT * y
    A = np.vstack((X[0], Y[0], dT[0], 
                   X[1], Y[1], dT[1])).T
    
    Hinv_AT = np.linalg.pinv(A.T @ A) @ A.T
    # print(Hinv_A.shape)
    
    for i in range(int(num_iters)):
        # Iw
        c = M @ C
        Iw = I.ev(c[1], c[0])
        
        # err 
        b = (Iw - T_).reshape(-1, 1)
        
        # dp
        dp = np.dot(Hinv_AT, b)
        # print(dp)
        dM = np.array([[1.+dp[0], dp[1], dp[2]], 
                       [dp[3], 1.+dp[4], dp[5]],
                       [0., 0., 1.]], dtype=np.float)
        M = M @ np.linalg.pinv(dM)
        # print(M)
        
        if np.linalg.norm(dp) <= threshold:
            break
        
    return M