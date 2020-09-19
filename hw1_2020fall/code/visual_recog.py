import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
import math
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap, norm=True):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    K = opts.K
    hist, _ = np.histogram(wordmap, bins=K)
    if norm:
        return hist / np.sum(hist)
    else:
        return hist
    
def compute_weight(l, L):
    if l == 0 or l == 1:
        return 2 ** (-L)
    return 2 ** (l - L - 1)

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
    K = opts.K
    L = opts.L
    hist_all = np.zeros((int(K*(4 ** L-1)/3)), dtype=float)
    
    # Get histograms of the finest layer
    i = 0
    y_cell = math.ceil(wordmap.shape[0] / (2 ** (L-1)))
    x_cell = math.ceil(wordmap.shape[1] / (2 ** (L-1)))
    for y in range(0, wordmap.shape[0], y_cell):
        max_y = wordmap.shape[0] if y > wordmap.shape[0] else y + y_cell
        for x in range(0, wordmap.shape[1], x_cell):
            max_x = wordmap.shape[1] if x > wordmap.shape[1] else x + x_cell
            # Get histogram here
            hist_all[i:i+K] = get_feature_from_wordmap(opts, 
                                                       wordmap[y:max_y, x:max_x], 
                                                       False)
            i += K
    # Normalize the layer 
    hist_all[:i] /= np.sum(hist_all[:i])

    # Get histograms of remaining layers
    j = i
    k = 0
    n_cells_1d_prev = 2**(L-1)
    for l in reversed(range(L-1)):
        n_cells_1d = 2**l
        for c in range(n_cells_1d * n_cells_1d):
            k1 = k + 2*K
            k2 = k + n_cells_1d_prev*K
            hist_all[j:j+K]  =  hist_all[k:k+K]
            hist_all[j:j+K] +=  hist_all[k+K:k1]
            hist_all[j:j+K] +=  hist_all[k2:k2+K]
            hist_all[j:j+K] +=  hist_all[k2+K:k1+n_cells_1d_prev*K]
            if (c+1) % n_cells_1d == 0:
                k = int(k1 + 4 * K * n_cells_1d / 2)
            else:
                k = int(k1)
            j+=K
        n_cells_1d_prev = n_cells_1d
    
    # Weight layers
    j = i
    # print(f"Layer {L-1} w {compute_weight(L-1, L-1)} range {0}-{K*4**(L-1)}")       
    for l in reversed(range(L-1)):
        num_cells = 2**l * 2**l
        hist_all[j:j+num_cells*K] *= compute_weight(l, L-1)
        # print(f"Layer {l} w {compute_weight(l, L-1)} range {j}-{j+num_cells*K}")
        j += num_cells*K
    hist_all[:4**(L-1)*K] *= compute_weight(L-1, L-1)
    
    # print(np.sum(hist_all), compute_weight(L-1, L-1))
    return hist_all
    
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    return get_feature_from_wordmap_SPM(opts, wordmap)

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    pass

    ## example code snippet to save the learned system
    # np.savez_compressed(join(out_dir, 'trained_system.npz'),
    #     features=features,
    #     labels=train_labels,
    #     dictionary=dictionary,
    #     SPM_layer_num=SPM_layer_num,
    # )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    sim = np.zero(histograms.shape[0])
    for i in range(histograms.shape[0]):
        sim[i] = 1 - np.sum(np.minimum(word_hist, histograms[i]))
    return sim    
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----
    pass

