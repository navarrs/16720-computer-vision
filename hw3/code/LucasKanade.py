import numpy as np
from scipy.interpolate import RectBivariateSpline
import numpy.matlib

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    assert It.shape == It1.shape, "Frames are different in shape"
    p = p0

    # create mesh of rect 
    rw, rh = int(rect[2]-rect[0]), int(rect[3]-rect[1])
    Xr, Yr = np.meshgrid(np.linspace(start=rect[0], stop=rect[2], num=rw, endpoint=False),
                         np.linspace(start=rect[1], stop=rect[3], num=rh, endpoint=False))
    
    x_, y_ = np.arange(It.shape[1]), np.arange(It.shape[0])
    T = RectBivariateSpline(y_, x_, z=It) # Template
    T_ = T.ev(Yr, Xr)
    I = RectBivariateSpline(y_, x_, z=It1) # Image
    
    dWdp = np.array([[1., 0.], [0., 1.]])
    dI = np.zeros((rw*rh, 2))
    
    for i in range(int(num_iters)):
        
        # I(W(X,p))
        Iw = I.ev(Yr + p[1], Xr + p[0])
        
        # T(X) - I(W(X,p))
        err_im = T_ - Iw
        b = err_im.reshape(-1, 1)
        
        # dI/dX'
        dIdy, dIdx = np.gradient(Iw)
        dI[:, 0] = dIdx.ravel()
        dI[:, 1] = dIdy.ravel()
        
        # dIdW
        A = dI @ dWdp
        H = A.T @ A 
        # A = np.linalg.pinv(H) @ A.T
        dp = np.linalg.inv(H) @ (A.T) @ b
        
        p[0] += dp[0]
        p[1] += dp[1]
               
        if np.linalg.norm(dp) <= threshold:
            break
            
    return p