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
    # print("Done")
    
    ## Q3.1 - Ablation results    
    # results = tune.get_results(opts, sorted=True)
    # tuning.display_results(results)
    
    ## Q3.2 - Custom 
    print("Q3.2 - Custom system")
    alpha = [125]
    filter_scales = [[1, 2]]
    K = [50]
    L = [3]
    
    # Evaluating default vs D
    D = [1, 10]
    # Evaluating default vs 0.8
    img_scale = [1.0, 0.8]
    tune.tune(alpha, filter_scales, K, L, D, img_scale)    
    results = tune.get_results(opts, sorted=True)
    tune.display_results(results)
    
if __name__ == '__main__':
    main()