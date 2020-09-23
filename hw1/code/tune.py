# 
# @date: Sept 20, 2020
#
# @brief Ablation tests for the visual recognition system
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from PIL import Image
from time import time

# Local includes
import util
import visual_words
import visual_recog
from opts import get_opts

def tune(alpha, filter_scales, K, L, D):
  """
    Tunes the recognition system 
    [input]
    * alpha: list of number of sampling pixels
    * filter scales: list of filter scales to extract responses
    * K: list of visual words
    * L: list of layers in SPM
    [saves]
    * dictionary, trained model and results per test 
  """
  opts = get_opts()
  n_cpu = util.get_num_CPU()
  OUT_DIR = opts.out_dir
  
  # Tests
  test = 0
  total_tests = len(L) * len(K) * len(filter_scales) \
              * len(alpha) * len(D)
              
  for d in D:
    opts.D = d
    for fs in filter_scales:
      opts.filter_scales = fs
      for a in alpha:
        opts.alpha = a
        for k in K:
          opts.K = k
          for l in L:
            opts.L = l
            test += 1
                
            # Create test directory 
            opts.out_dir = OUT_DIR + f"/test_l-{l}_k-{k}_fs-{len(fs)}_a-{a}_d-{d}"
            if not os.path.exists(opts.out_dir):
              os.mkdir(opts.out_dir)
                  
            print(f"TEST [{test}/{total_tests}]: L-{l}_K-{k}_fs-{len(fs)}_a-{a}_d-{d}")
              
            # Dictionary  
            if not os.path.exists(join(opts.out_dir, "dictionary.npy")):
              print("\tBuilding Dictionary")
              start = time()
              visual_words.compute_dictionary(opts, n_worker=n_cpu)
              print(f"Time  {(time() - start) / 60.0}")
            else: 
              print("\tDictionary exists")
              
            # Train
            if not os.path.exists(join(opts.out_dir, "trained_system.npz")):
              print("\tBuilding Recognition System")
              start = time()
              visual_recog.build_recognition_system(opts, n_worker=n_cpu)
              print(f"Time  {(time() - start) / 60.0}")
            else:
              print("\tRecognition system exists")
              
            # Test
            if not os.path.exists(join(opts.out_dir, "accuracy.txt")) or \
              not os.path.exists(join(opts.out_dir, "confmat.csv")) or \
              not os.path.exists(join(opts.out_dir, "model.npz")):  
              # Test
              print("\tEvaluation")
              start = time()
              conf, acc = visual_recog.evaluate_recognition_system(opts, 
                                                                   n_worker=n_cpu)
              print(f"Time  {(time() - start) / 60.0}")
              
              # Results
              print(f"Confusion Matrix\n{conf}\n Accuracy: {acc}")
              np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, 
                        fmt='%d', delimiter=',')
              np.savetxt(join(opts.out_dir, 'accuracy.txt'), [acc], fmt='%g')
              np.savez_compressed(join(opts.out_dir, "model.npz"), 
                filter_scales=fs, K=k, L=l, alpha=a, acc=acc, conf_mat=conf)
            else:
              print("\tEvaluation exists")

def get_results(opts, sorted=True):
  """
    Reads all output directories and returns the models
  """
  def take_acc(v):
    return v["acc"]
  
  dirs = glob(opts.out_dir + "/test*")
  models = []
  for i, d in enumerate(dirs):
    model_file = join(d, "model.npz")
    if os.path.exists(model_file):
      models.append(np.load(model_file, allow_pickle=True))
    else:
      print(f"Could not find results in {d}")
  
  if sorted:
    models.sort(key=take_acc)
    
  return models

def display_results(results):
  print(f"Scales\tAlpha\tK\tL\tAcc")
  for result in results:
    fs = result["filter_scales"]
    k = result["K"]
    l = result["L"]
    acc = result["acc"]
    alpha = result["alpha"]
    print(f"{fs}\t{alpha}\t{k}\t{l}\t{acc}")