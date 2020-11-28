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
    from scipy.sparse import linalg

    u, s, vt = linalg.svds(I, k=3)
    # print(u.shape, vt.shape)
    B = vt[:3, :]
    L = u[:, :3].T
    return B, L


if __name__ == "__main__":

    # --------------------------------------------------------------------------
    # Q2.A
    # --------------------------------------------------------------------------
    I, L, s = loadData()
    print("Q1C -- I:{} L:{} s:{}".format(I.shape, L.shape, s))
    # print(L)

    # --------------------------------------------------------------------------
    # Q2.B
    # --------------------------------------------------------------------------
    B, L = estimatePseudonormalsUncalibrated(I)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    # plt.imshow(albedoIm, cmap='gray')
    # plt.savefig("../out/q2/albedo_before.png")
    # plt.close()
    # plt.imshow(normalIm, cmap='rainbow')
    # plt.savefig("../out/q2/normals_before.png")
    # plt.close()
    # print(L)

    # --------------------------------------------------------------------------
    # Q2.D
    # --------------------------------------------------------------------------
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # --------------------------------------------------------------------------
    # Q2.E
    # --------------------------------------------------------------------------
    normals = enforceIntegrability(B, s)
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # --------------------------------------------------------------------------
    # Q2.F
    # --------------------------------------------------------------------------
    exp = [[0, 0, 0.5], [0, 0, 1.5],  # changing lambda
           [0.5, 0, 1], [1.5, 0, 1],  # changing mu
           [0, 0.5, 1], [0, 1.5, 1]]  # changing v
    for e in exp:
        G = np.array([[1, 0, 0], [0, 1, 0], e])
        B_ = np.linalg.inv(G).T @ B
        normals = enforceIntegrability(B_, s)
        surface = estimateShape(normals, s)
        plotSurface(surface)

    # --------------------------------------------------------------------------
    # Q2.G
    # --------------------------------------------------------------------------
    G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.001]])
    B_ = np.linalg.inv(G).T @ B
    normals = enforceIntegrability(B_, s)
    surface = estimateShape(normals, s)
    plotSurface(surface)
