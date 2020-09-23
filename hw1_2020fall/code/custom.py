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

def main():
    opts = get_opts()
    
    ## Q3.1 - Hyperparameter tunning 
    # print("Q3.1 - Hyper Parameter tunning")
    # alpha = [25, 125]
    # filter_scales = [[1, 2], [1, 2, 4]]
    # K = [10, 50]
    # L = [3, 2, 1]
    # tune.tune(alpha, filter_scales, K, L)  
    # results = tune.get_results(opts, sorted=True)
    # tuning.display_results(results)
    
    ## Q3.2 - Custom 
    print("Q3.2 - Custom system with default parameters")
    alpha = [25]
    filter_scales = [[1, 2]]
    K = [10]
    L = [1]
    # Evaluating default vs D
    D = [1, 5]
    tune.tune(alpha, filter_scales, K, L, D)    
    results = tune.get_results(opts, sorted=True)
    tune.display_results(results)
    
    print("Q3.2 - Custom system with best parameters")
    alpha = [125]
    filter_scales = [[1, 2]]
    K = [50]
    L = [3]
    
    # Evaluating default vs D
    D = [1, 10]
    # Evaluating default vs 0.8
    tune.tune(alpha, filter_scales, K, L, D)    
    results = tune.get_results(opts, sorted=True)
    tune.display_results(results)
    
if __name__ == '__main__':
    main()