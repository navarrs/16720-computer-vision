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
    p = np.zeros(6)
    
    H, W = It.shape[0], It.shape[1]
    x_, y_ = np.arange(W), np.arange(H)
    
    T = RectBivariateSpline(y_, x_, It)
    I = RectBivariateSpline(y_, x_, It1)

    # homogeneous coords
    c = np.array([[0.0, 0.0, 1.0],
                  [  W,   H, 1.0]]).T
    
    for i in range(int(num_iters)):

        M = np.array([[1.0 + p[0], p[1], p[2]],
                      [p[3], 1.0 + p[4], p[5]], 
                      [0.0, 0.0, 1.0]])
        
        # Region shared
        c_ = np.matmul(M, c)
        x1 = c_[0, 0] #if c_[0, 0] > 0.0 else 0.0
        y1 = c_[1, 0] #if c_[1, 0] > 0.0 else 0.0
        x2 = c_[0, 1] #if c_[0, 1] < float(W[0]) else float(W[0])
        y2 = c_[1, 1] #if c_[1, 1] < float(H[0]) else float(H[0])

        # print(x1, y1, x2, y2)
        
        X, Y = np.meshgrid(np.linspace(x1, x2, It.shape[1]),
                           np.linspace(y1, y2, It.shape[0]))

        # Iw
        Iw = I.ev(Y, X)
        T_ = T.ev(Y, X)
        
        # Tx - Iw
        err = T_ - Iw
        b = err.reshape(-1, 1)

        # dI
        dIdx, dIdy = np.gradient(Iw)
        dI = np.vstack((dIdx.ravel(), dIdy.ravel())).T

        yy, xx = len(Y), len(X)
        A = np.zeros((yy*xx, 6))
        
        k = 0
        for i in range(yy):
            for j in range(xx):
                # dWdp = | x 0 y 0 1 0 |
                #        | 0 x x y 0 1 |
                dWdp = np.array([[j, 0, i, 0, 1, 0],
                                 [0, j, 0, i, 0, 1]])
                
                # dIdW = dI @ dWdp
                A[k] = dI[k] @ dWdp
                k += 1

        dp = np.linalg.pinv(A) @ b

        p[0] += dp[0]
        p[1] += dp[1]
        p[2] += dp[2]
        p[3] += dp[3]
        p[4] += dp[4]
        p[5] += dp[5]
        
        if np.linalg.norm(dp) <= threshold:
            break

    M = np.array([[1.0 + p[0], p[1], p[2]],
                [p[3], 1.0 + p[4], p[5]],
                [0, 0, 1]])
    return M