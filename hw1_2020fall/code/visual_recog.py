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
            hist = get_feature_from_wordmap(opts, wordmap[y:max_y, x:max_x], False)
            hist_all[i:i+K] = hist
            i += K
    # Normalize the layer 
    hist_all[:i] /= np.sum(hist_all[:i])

    # Add up from finest to largest
    j = i
    k = 0
    for l in reversed(range(L-1)):
        print(f"layer {l}")
        nc = 2**(l+1)
        n = 2**l
        num_cells = 2**l * 2**l
        for c in range(num_cells):
            k_ = k + 2*K
            print(f"Cell {c} gets {k}-{k + 2*K} and {k_ + (nc-2)*K}-{k_ + nc*K}")
            hist_all[j:j+K]  =  hist_all[k:k + K]
            hist_all[j:j+K] +=  hist_all[k+K:k_]
            hist_all[j:j+K] +=  hist_all[k_ + (nc-2)*K:k_ + (nc-2)*K+K]
            hist_all[j:j+K] +=  hist_all[k_ + (nc-2)*K+K:k_ + nc*K]
        
            if (c+1) % n == 0:
                k = int(k_ + 4 * K * n / 2)
            else:
                k = int(k_)
            j+=K
    print(hist_all, np.sum(hist_all))   
    # k = 0
    # for l in reversed(range(L-1)):
    #     num_cells = 2**l * 2**l
    #     print(f"Layer {l}")
    #     for c in range(num_cells):
    #         k_max = k + 4 * K
    #         print(f"Cell {c} gets from {k} to {k_max} into j {j}")
    #         hist_all[j:j+K] = np.sum(hist_all[k:k_max])
    #         k = k_max
    #         j += K
    # print(np.sum(hist_all))
            
    # j = i
    # for l in reversed(range(L-1)):
    #     num_cells = 2**l * 2**l
    #     mod = 4 * (L-1-l)
    #     for c in range(num_cells):
    #         for sub_c in range(4):
    #             start_i = ((i//K) % mod) * K
    #             print(f"mod {mod} start {start_i}")     
    #             hist_all[j:j+K] += hist_all[start_i:start_i+K]
    #             i += K
    #         hist_all[j:j+K] /= np.sum(hist_all[j:j+K])
    #         j += K
    #print(hist_all, np.sum(hist_all))
    
    return 0
    
    
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

    # ----- TODO -----
    pass

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

    # ----- TODO -----
    pass    
    
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

