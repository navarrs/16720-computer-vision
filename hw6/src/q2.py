# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface
from utils import enforceIntegrability
from matplotlib import pyplot as plt


def estimatePseudonormalsUncalibrated(I):
    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions.

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """
    u, s, vt = np.linalg.svd(I, full_matrices=False)
    s = np.sqrt(np.diag(s[:3]))
    B = s @ vt[:3, :]
    L = (u[:, :3] @ s).T
    return B, L


if __name__ == "__main__":

    # --------------------------------------------------------------------------
    # Q2.A
    # --------------------------------------------------------------------------
    I, L_truth, s = loadData()
    print("Q2A -- I:{} L:{} s:{}".format(I.shape, L_truth.shape, s))
    # print(L)

    # --------------------------------------------------------------------------
    # Q2.B
    # --------------------------------------------------------------------------
    B, L = estimatePseudonormalsUncalibrated(I)
    print("Q2B-- L:{} B:{} s:{}".format(L.shape, B.shape, s))
    albedos, normals = estimateAlbedosNormals(B)
    normals[-1, :] = -normals[-1, :]
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    
    # plt.imshow(albedoIm, cmap='gray')
    # plt.savefig("../out/q2/albedo.png")
    # plt.show()
    # plt.close()
    # plt.imshow(normalIm, cmap='rainbow')
    # plt.savefig("../out/q2/normals.png")
    # plt.show()
    # plt.close()
    # print(L)
    
    # --------------------------------------------------------------------------
    # Q2.C
    # --------------------------------------------------------------------------
    # print("Q2C --------------------------------------------------")
    # fig, axs = plt.subplots(2)
    # fig.suptitle('Light Comparison')
    # axs[0].imshow(L_truth, cmap='hot')
    # axs[0].set_title('Ground truth light')
    # axs[1].imshow(L, cmap='hot')
    # axs[1].set_title('Estimated light')
    # plt.show()

    # --------------------------------------------------------------------------
    # Q2.D
    # --------------------------------------------------------------------------
    print("Q2D --------------------------------------------------")
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # --------------------------------------------------------------------------
    # Q2.E
    # --------------------------------------------------------------------------
    print("Q2E --------------------------------------------------")
    normals = enforceIntegrability(B, s)
    normals[-1, :] = -normals[-1, :]
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # --------------------------------------------------------------------------
    # Q2.F
    # --------------------------------------------------------------------------
    print("Q2F --------------------------------------------------")
    exp = [
        #[0, 0, -1], [0, 0, 1.5],  # changing lambda
        #[-2.5, 0, 1], [2.5, 0, 1],  # changing mu
        #[0, -2.5, 1], [0, 2.5, 1] # changing v
    ]  
    for e in exp:
        G = np.array([[1, 0, 0], [0, 1, 0], e])
        B_ = np.linalg.inv(G).T @ B
        normals = enforceIntegrability(B_, s)
        normals[-1, :] = -normals[-1, :]
        surface = estimateShape(normals, s)
        plotSurface(surface)

    # --------------------------------------------------------------------------
    # Q2.G
    # --------------------------------------------------------------------------
    print("Q2G --------------------------------------------------")
    G = np.array([[1, 0, 0], [0, 1, 0], [1.5, 0, 1.5]])
    B_ = np.linalg.inv(G).T @ B
    normals = enforceIntegrability(B_, s)
    normals[-1, :] = -normals[-1, :]
    surface = estimateShape(normals, s)
    plotSurface(surface)
