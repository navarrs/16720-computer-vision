import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import scipy.signal
import skimage.color
from enum import Enum

class Filter(Enum):
    " Enum containing the types of filters available"
    GAUSSIAN = 1
    LOG = 2
    DOG_X = 3
    DOG_Y = 4

def gaussian(ksize, sigma=1.0):
    kernel = np.zeros((ksize, ksize))
    for i in range(ksize):
        x = i - (ksize-1) / 2
        for j in range(ksize):
            y = j - (ksize-1) / 2
            kernel[i, j] = np.exp(-(x**2+y**2/(2*sigma**2)))
    return (1/(2*np.pi*sigma**2)) * kernel

def laplacian_of_gaussian(ksize, sigma=1.0):
    kernel = np.zeros((ksize, ksize))
    for i in range(ksize):
        x = i - (ksize-1) / 2
        for j in range(ksize):
            y = j - (ksize-1) / 2
            a = (x ** 2 + y ** 2) / 2.0 * sigma ** 2
            kernel[i, j] = -(1/(np.pi * sigma ** 4)) * (1 - a) * np.exp(-a)
    return kernel

def derivative_of_gaussian_x(ksize, sigma1=1.0, sigma2=2.0):
    kernel = np.zeros((ksize, ksize))
    sigma1_2sq = 2 * sigma1 ** 2
    sigma2_2sq = 2 * sigma2 ** 2
    for i in range(ksize):
        x = (i - (ksize-1) / 2) ** 2
        for j in range(ksize):
            kernel[i, j] = 1/sigma1 * np.exp(-x/sigma1_2sq) - 1/sigma2 * np.exp(-x/sigma2_2sq)
    return 1 / np.sqrt(2 * np.pi) * kernel

def derivative_of_gaussian_y(ksize, sigma1=1.0, sigma2=2.0):
    kernel = np.zeros((ksize, ksize))
    sigma1_2sq = 2 * sigma1 ** 2
    sigma2_2sq = 2 * sigma2 ** 2
    for i in range(ksize):
        for j in range(ksize):
            y = (j - (ksize-1) / 2) ** 2
            kernel[i, j] = 1/sigma1 * np.exp(-y/sigma1_2sq) - 1/sigma2 * np.exp(-y/sigma2_2sq)
    return 1 / np.sqrt(2 * np.pi) * kernel

def get_kernel(ktype, ksize, opts):
    if ktype == Filter.GAUSSIAN:
        return gaussian(ksize, opts.sigma_gaussian)
    elif ktype == Filter.LOG:
        return laplacian_of_gaussian(ksize, opts.sigma_log) 
    elif ktype == Filter.DOG_X:
        return derivative_of_gaussian_x(ksize, opts.sigma1_dogx, opts.sigma2_dogx)
    elif ktype == Filter.DOG_Y:
        return derivative_of_gaussian_y(ksize, opts.sigma1_dogy, opts.sigma2_dogy)

def convolve2d(img, kernel):
    response = np.zeros(img.shape)
    scipy.ndimage.convolve(img, kernel, response)
    return response


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    # Check if the image is floating point type and in range [0, 1]
    # Solve this later
    # if img.dtype != np.float32: 
    #     if img.dtype.kind == 'u':
    #         img = img.astype(np.float32) / np.iinfo(img.dtype).max
    #     else:
    #         print("Unsupported conversion")
    #         return 0

    # Check if image is grayscale, if so, convert it into three channel image
    # if len(img.shape) == 2:
    #     img = np.array([img, img, img]) 
    #     img = np.moveaxis(img, 0, 2)

    # Convert image from RGB to Lab
    img = skimage.color.rgb2lab(img)

    filter_scales = opts.filter_scales
    filter_responses = np.zeros(
        (img.shape[0], img.shape[1], len(Filter) * len(filter_scales) * img.shape[2]))
    i = 0
    for fscale in filter_scales:
        for ftype in Filter:
            # Compute kernel 
            kernel = get_kernel(ftype, fscale, opts)
            print(f"{ftype} with scale {fscale} is \n{kernel}")
            for c in range(img.shape[2]):
                filter_responses[:, :, i] = convolve2d(img[:, :, c], kernel)
                i += 1
    return filter_responses

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    pass

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    pass

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    pass

