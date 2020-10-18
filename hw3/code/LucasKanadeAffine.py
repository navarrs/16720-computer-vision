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

    # put your implementation here
    M = np.eye(3)
    p = np.zeros(6)

    x, y = np.arange(It.shape[1]), np.arange(It.shape[0])
    T = RectBivariateSpline(y, x, z=It)
    I = RectBivariateSpline(y, x, z=It1)  # Image

    # Homogeneous coords
    X, Y = np.meshgrid(x, y)
    h = np.zeros((3, len(x)*len(y)))
    h[0, :] = X.reshape(len(x)*len(y))
    h[1, :] = Y.reshape(len(x)*len(y))
    h[2, :] = 1
    # print(h.shape)

    for i in range(int(num_iters)):
        M = np.array([[1+p[0],   p[1], p[3]], 
                      [  p[3], 1+p[4], p[5]], 
                      [     0,      0,    1]])
        c = np.matmul(M, h)
        # print(c.shape)
        
        # Find pixels that would be common to warped version of It1
        x_, y_ = c[0], c[1]
        # print(x_, y_)
        # print(np.where(x_ <= len(x)))
        x_, y_ = x_[np.where(x_ <= len(x))], y_[np.where(y_ <= len(y))]
        # print(x_, y_)

        # I(W(x:p))
        I_ = I.ev(x_, y_)
        T_ = T.ev(x_, y_)
        
        # T(X) - I(W(X,p))
        err_im = T_ - I_
        b = err_im.reshape(-1, 1)

        # dI
        dIdx = I.ev(x_, y_, dy=1).ravel()
        dIdy = I.ev(x_, y_, dx=1).ravel()
        dI = np.matrix([dIdx, dIdy]).T

        A = np.zeros((len(dIdx), 6))

        for n in range(len(dI)):
            # dWdp = | x 0 y 0 1 0 |
            #        | 0 x 0 y 0 1 |
            dWdp = np.matrix([[x_[n], 0., y_[n], 0., 1., 0.],
                              [0., x_[n], 0., y_[n], 0., 1.]])
            A[n] = dI[n] @ dWdp
            
        dp = np.dot(np.linalg.pinv(A), b)
        p[0] += dp[0]
        p[1] += dp[1]
        p[2] += dp[2]
        p[3] += dp[3]
        p[4] += dp[4]
        p[5] += dp[5]
        
        if np.linalg.norm(dp) <= threshold:
            break
    
    M = np.matrix([[1+p[0], p[1], p[3]], 
                   [p[3], 1+p[4], p[5]], 
                   [0, 0, 1]])
    
    return M
