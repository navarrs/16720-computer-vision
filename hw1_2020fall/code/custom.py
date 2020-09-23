# 
# @date: Sept 20, 2020
#
# @brief Homework 1
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import os
from PIL import Image
from time import time
import os, math, multiprocessing
from itertools import repeat

# Local includes
import util
import visual_words
import visual_recog
from opts import get_opts
import tune

def boost_eval(opts, model_paths, n_worker=1):
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
    pass
    # trained_systems = []
    # trained_features = []
    # trained_labels = []
    # model_params = []
    # data_dir = opts.data_dir
    
    # for i, mp in enumerate(model_paths):
    #   trained_systems.append(np.load(join(mp, 'trained_system.npz')))
    #   model_params.append(np.load(join(mp, 'model.npz')))
    
    # test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    # test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
        
    # Parallel 
    # features = []
    # for i in range(len(model_paths)):
    #   # filter_scales=fs, K=k, L=l, alpha=a
    #   opts.K = model_params[i]["K"]
    #   opts.L = int(model_params[i]["L"])
    #   opts.alpha = model_params[i]["alpha"]
    #   opts.filter_scales = model_params[i]["filter_scales"]
    #   print(opts.filter_scales, opts.K, opts.L, opts.alpha)
    #   dictionary = trained_systems[i]['dictionary']
    #   f = [visual_recog.get_image_feature(opts, tfile, dictionary) for tfile in test_files]
    #   features.append(f)
    #   print("Done with features")
    # save features to file 
    
      # trained_features = trained_systems[i]['features']
      # trained_labels = trained_systems[i]['labels']
    # conf = np.zeros((8, 8), dtype=float)
    # N = len(test_labels)
    # for i in range(N): 
    #   gt_class = test_labels[i]
    #   dists = [visual_recog.distance_to_set(f[i], trained_features) for f in features]
    #   print(np.asarray(dists).shapes)
      
    #   # est_class = trained_labels[np.argmin(distance_to_set(features[i], 
    #   #                                                           trained_features))]
    #   # conf[gt_class, est_class] += 1
    #   # Write result 
    #   # print(f" GT: {gt_class} EST: {est_class} progress {100*i//N}") 
    # return conf, np.trace(conf / np.sum(conf))

def main():
    opts = get_opts()
    
    ## Q3.1 - Hyperparameter tunning 
    # print("Q3.1 - Hyper Parameter tunning")
    # alpha = [25, 125]
    # filter_scales = [[1, 2], [1, 2, 4]]
    # K = [10, 50]
    # L = [3, 2, 1]
    # tune.tune(alpha, filter_scales, K, L)
    # print("Done")
    
    ## Q3.1 - Ablation results    
    # results = tune.get_results(opts, sorted=True)
    # tuning.display_results(results)
    
    model_paths = [
      "../out_/test_l-3_k-50_fs-2_a-125",
      "../out_/test_l-2_k-30_fs-3_a-25",
    ]
    n_cpu = util.get_num_CPU()
    boost_eval(opts, model_paths, n_cpu)
    
    ## Q3.2 - Custom 
    # print("Q3.2 - Custom system with default parameters")
    # alpha = [25]
    # filter_scales = [[1, 2]]
    # K = [10]
    # L = [1]
    # # Evaluating default vs D
    # D = [1, 5]
    # # Evaluating default vs 0.8
    # tune.tune(alpha, filter_scales, K, L, D)    
    # results = tune.get_results(opts, sorted=True)
    # tune.display_results(results)
    
    # print("Q3.2 - Custom system with best parameters")
    # alpha = [125]
    # filter_scales = [[1, 2]]
    # K = [50]
    # L = [3]
    
    # # Evaluating default vs D
    # D = [1, 10]
    # # Evaluating default vs 0.8
    # tune.tune(alpha, filter_scales, K, L, D)    
    # results = tune.get_results(opts, sorted=True)
    # tune.display_results(results)
    
if __name__ == '__main__':
    main()