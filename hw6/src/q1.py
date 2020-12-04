# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

# Imports
import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2xyz
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.sparse import linalg
from utils import integrateFrankot

def renderNDotLSphere(center, rad, light, pxSize, res):
    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centered on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the sphere in an array of size (3,)

    rad : float
        The radius of the sphere

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the sphere
    """
    pxSize *= 100
    radpx = rad / pxSize
    Y, X = np.ogrid[:res[1], :res[0]]
    cx, cy = (res[0]-center[0])//2, (res[1]-center[1])//2
    image_bools = np.sqrt((X - cx)**2 + (Y - cy)**2) < radpx
    image = np.zeros(image_bools.shape)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image_bools[i, j]:
                # pixel to meter
                y = i * pxSize
                x = j * pxSize
                # compute depth value
                z = np.sqrt(abs(rad**2 - x**2 - y**2))
                # compute normal at x, y, z
                n = np.array([x, y, z]) 
                n = n / np.linalg.norm(n)
                # compute shading
                image[i, j] = max(0, np.dot(n, light))
            else:
                image[i, j] = 0.3 # an arbitrary background value
    
    plt.imshow(image, origin='lower')
    # plt.savefig('../out/q1/sphere-light-{:.2f}_{:.2f}_{:.2f}.png'
    #             .format(light[0], light[1], light[2]), 
    #             bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    return image


def loadData(path = "../data/"):
    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    I = []
    for i in range(7):
        image = imread(os.path.join(path, f'input_{i+1}.tif'))
        if image.dtype != np.uint16:
            image = image.astype(np.uint16)
            
        # Luminance channel is Y
        image = rgb2xyz(image)[:, :, 1] 
        s = image.shape
        I.append(image.flatten())
    
    I = np.asarray(I)
    # print(I.dtype)
    L = np.load(os.path.join(path, "sources.npy")).T
    # print(I.shape, L.shape)
    return I, L, s


def estimatePseudonormalsCalibrated(I, L):
    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    A = L.T
    
    B = np.zeros((3, I.shape[1]))
    for i in range(I.shape[1]):
        y = I[:, i]
        B[:, i] = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # print(f"y: {y.shape} {A.shape} {L.shape}")
    # B = np.linalg.inv(L @ L.T) @ L @ I
    return B


def estimateAlbedosNormals(B):
    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''
    albedos = np.linalg.norm(B, axis=0)
    normals = B / albedos
    normals[-1, :] = -normals[-1, :]
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):
    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """
    albedoIm = albedos.reshape(s)
    normalIm = normals.T.reshape(s[0], s[1], 3)
    
    plt.imshow(albedoIm, cmap='gray')
    # plt.savefig("../out/q2/albedo.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    
    plt.imshow(normalIm, cmap='rainbow')
    # plt.savefig("../out/q2/normal.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

    return albedoIm, normalIm


def estimateShape(normals, s):
    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point
    """
    xz = np.zeros(s)
    yz = np.zeros(s)
    normals = normals.T.reshape(s[0], s[1], 3)
    for i in range(s[0]):
        for j in range(s[1]):
            xz[i, j] = - normals[i,j, 0] / normals[i,j, 2]
            yz[i, j] = - normals[i,j, 1] / normals[i,j, 2]
    surface = integrateFrankot(xz, yz)
    return surface


def plotSurface(surface):
    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(0, surface.shape[1], 1)
    Y = np.arange(0, surface.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    # print(X.shape, Y.shape)

    # Plot the surface.
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    surf = ax.plot_surface(X, Y, surface, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    plt.show()
    


if __name__ == '__main__':

    
    # --------------------------------------------------------------------------
    # Q1.B
    # --------------------------------------------------------------------------
    print(f"Q1B", "-"*50)
    center = np.array([0, 0, 10])
    rad = 0.75
    light = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1]]) / np.sqrt(3)
    pxSize = 7e-6
    # res = [100, 100]
    res = [3840, 2160]
    for i in range(len(light)):
        renderNDotLSphere(center, rad, light[i], pxSize, res)
    
    # --------------------------------------------------------------------------
    # Q1.C
    # --------------------------------------------------------------------------
    I, L, s = loadData()
    print("Q1C -- I:{} L:{} s:{}".format(I.shape, L.shape, s))
    
    # --------------------------------------------------------------------------
    # Q1.D
    # --------------------------------------------------------------------------
    _, S, _ = np.linalg.svd(I, full_matrices=False)
    print(f"Q1D Numpy -- Singular values: {S}")
    # _, S, _ = linalg.svds(I)
    # print(f"Q1D Scipy -- Singular values: {S}")
    
    # --------------------------------------------------------------------------
    # Q1.E
    # --------------------------------------------------------------------------
    B = estimatePseudonormalsCalibrated(I, L)
    print(f"Q1E -- B:{B.shape}")
    
    # --------------------------------------------------------------------------
    # Q1.F
    # --------------------------------------------------------------------------
    albedos, normals = estimateAlbedosNormals(B)
    displayAlbedosNormals(albedos, normals, s)
    print(f"Q1F -- albedos: {albedos.shape} normals: {normals.shape}")
    
    # --------------------------------------------------------------------------
    # Q1.H
    # --------------------------------------------------------------------------
    surface = estimateShape(normals, s)
    plotSurface(surface)