import os, math, multiprocessing
import itertools
from os.path import join
from copy import copy
from itertools import repeat
import numpy as np
import math
from PIL import Image
import visual_words
from enum import Enum

class Class(Enum):
    AQUARIUM = 0
    DESERT = 1
    HIGHWAY = 2
    KITCHEN = 3
    LAUNDROMAT = 4
    PARK = 5
    WATERFALL = 6
    WINDMILL = 7
    
################################################################################
# Q2.1
def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    K = opts.K
    # Reference: https://stackoverflow.com/questions/18082536/
    #            numpy-histogram-normalized-with-specified-edges-python
    bins = np.linspace(0, K, K + 1)
    hist, _ = np.histogram(wordmap, bins=bins, density=True)
    return hist
    # hist, _ = np.histogram(wordmap, bins=K)
    # if norm:
    #     return hist / np.sum(hist) 
    # return hist

################################################################################
# Q2.1  
def compute_weight(l, L):
    '''
        Helper method to compute weight value for a layer in SPM
        [input]
        * l: layer
        * L: num of layers
        [output]
        * weights
    '''
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
    j = 0
    i = 0
    for l in reversed(range(L)):   
        y_cell = math.ceil(wordmap.shape[0] / (2 **l))
        x_cell = math.ceil(wordmap.shape[1] / (2 **l))
        w = compute_weight(l, L-1)
        #print(y_cell, x_cell, w)
        
        for y in range(0, wordmap.shape[0], y_cell):
            max_y = wordmap.shape[0] if y > wordmap.shape[0] else y + y_cell
            for x in range(0, wordmap.shape[1], x_cell):
                max_x = wordmap.shape[1] if x > wordmap.shape[1] else x + x_cell  
                # print(f"y ({y},{max_y}), x ({x},{max_x}), cell({y_cell},{x_cell})")
                cell_word = wordmap[y:max_y, x:max_x]
                hist = get_feature_from_wordmap(opts, cell_word)
                hist_all[i:i+K] = hist
                i += K
        hist_all[j:i] /= np.sum(hist_all[j:i])
        hist_all[j:i] *= w
        j = i       
    # print(hist_all.shape, np.sum(hist_all))
    return hist_all

################################################################################
# Q2.3  
def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    # Sequential 
    # sim = np.zeros(histograms.shape[0])
    # for i in range(histograms.shape[0]):
    #     sim[i] = 1 - np.sum(np.minimum(word_hist, histograms[i]))
    
    # Parallel
    sim = np.sum(np.minimum(word_hist, histograms), axis=1)
    return 1. - sim
        

################################################################################
# Q2.4
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
    #print(img_path)
    img = Image.open(join(opts.data_dir, img_path))
    img = np.array(img).astype(np.float32)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    return get_feature_from_wordmap_SPM(opts, wordmap)

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from 
    all training images.

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
    K = opts.K
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    
    # Sequential 
    # N = len(train_labels)
    # features = np.zeros((N, int(K*(4 ** SPM_layer_num-1)/3)), dtype=float)
    # for i in range(N):
    #     features[i] = get_image_feature(opts, train_files[i], dictionary)
    
    # Parallel
    pool = multiprocessing.Pool(n_worker)
    features = pool.starmap(get_image_feature, 
                            zip(repeat(opts), train_files, repeat(dictionary)))
        
    # Save the model
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=np.array(features),
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

################################################################################
# Q2.5  
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the 
    confusion matrix and accuracy 

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
    trained_features = trained_system['features']
    trained_labels = trained_system['labels']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
    
    # Sequential 
    # features = [get_image_feature(opts, tfile, dictionary) for tfile in test_files]
        
    # Parallel 
    pool = multiprocessing.Pool(n_worker)
    features = pool.starmap(get_image_feature, 
                            zip(repeat(opts), test_files, repeat(dictionary)))
    
    # Eval 
    conf = np.zeros((8, 8), dtype=float)
    N = len(test_labels)
    predfile = join(opts.out_dir, "pred.txt")
    if os.path.exists(predfile):
        os.remove(predfile)
        
    pred = open(predfile, "a")
    
    if opts.D == 1:
        for i in range(N): 
            gt_class = test_labels[i]
            est_class = trained_labels[np.argmin(distance_to_set(features[i], 
                                                                trained_features))]
            conf[gt_class, est_class] += 1
            # Write result 
            # print(f" GT: {gt_class} EST: {est_class} progress {100*i//N}") 
            pred.write(f"{gt_class},{est_class},{test_files[i]}\n")
    else:
        for i in range(N):
            gt_class = test_labels[i]
            idx = np.argsort(distance_to_set(features[i], trained_features))[:opts.D]
            est_class = np.argmax(np.bincount(trained_labels[idx]))
            conf[gt_class, est_class] += 1
            # Write result 
            # print(f" GT: {gt_class} EST: {est_class} progress {100*i//N}")    
            pred.write(f"{gt_class},{est_class},{test_files[i]}\n")
    pred.close()
    return conf, np.trace(conf / np.sum(conf))

################################################################################
# Q2.6 
def get_common_fails(opts, save_errors=False):
    """
        Computes a list of the classes with highest mis-classifications
        [input]
        * opts
        * save_errors: if enabled saves misclassified images into a prediction directory
        [output]
        * a list containing [ground-truth class, predicted class, % missclassifications]
        
    """
    conf_mat = np.loadtxt(join(opts.out_dir, "confmat.csv"), delimiter=",")
    conf_mat /= np.sum(conf_mat, axis=1)
    print(conf_mat)
    np.fill_diagonal(conf_mat, 0)
    
    max_errors = np.argwhere(conf_mat >= opts.thresh_err)
    
    # Extract the most common errors
    if save_errors:
        pred_dir = join(opts.out_dir, "preds")
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
            
        preds_file = open(join(opts.out_dir, "pred.txt"), "r")
        for line in preds_file.readlines():
            pred = line.split(',')
            gt = pred[0]
            est = pred[1]
            if gt != est and [int(gt), int(est)] in max_errors:            
                gt = Class(int(gt)).name
                est = Class(int(est)).name
                img_path = pred[2].replace('\n', '')
                # print(f"{img_path} is {gt} but predicted as {est}")
                img = Image.open(join(opts.data_dir, img_path))
                img_name = img_path.split("/")[-1].split(".")[0] + f"_g-{gt}_e-{est}.png"
                img.save(join(pred_dir, img_name), "PNG")
        preds_file.close()
        
    return [[Class(e[0]), Class(e[1]), conf_mat[e[0],e[1]]] for e in max_errors]