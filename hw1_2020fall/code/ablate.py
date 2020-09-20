from os.path import join
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from time import time

# Local includes
import util
import visual_words
import visual_recog
from opts import get_opts

def main():
  opts = get_opts()
  n_cpu = util.get_num_CPU()
  OUT_DIR = opts.out_dir
    
  # Hyperparameters for the ablation study
  alpha = [25, 75, 125]
  filter_scales = [[1, 2], [1, 2, 4]]
  K = [10, 20]
  L = [1, 2, 3]
    
  # Tests
  test = 0
  total_tests = len(L) * len(K) * len(filter_scales) * len(alpha)
  for a in alpha:
    opts.alpha = a
    for fs in filter_scales:
      opts.filter_scales = fs
      for k in K:
        opts.K = k
        for l in L:
          opts.L = l
          test += 1
            
          # Create test directory 
          opts.out_dir = OUT_DIR + f"/test_l-{l}_k-{k}_fs-{len(fs)}_a-{a}"
          if not os.path.exists(opts.out_dir):
            os.mkdir(opts.out_dir)
          else:
            continue
              
          # Dictionary  
          print(F"TEST [{test}/{total_tests}]: L-{l}_K-{k}_fs-{len(fs)}_a-{a}")
          print("\tBuilding Dictionary")
          start = time()
          visual_words.compute_dictionary(opts, n_worker=n_cpu)
          print(f"Time  {(time() - start) / 60.0}")
            
          # Train
          print("\tBuilding Recognition System")
          start = time()
          visual_recog.build_recognition_system(opts, n_worker=n_cpu)
          print(f"Time  {(time() - start) / 60.0}")
            
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
          hyper_params = {
            "filter_scales" : opts.filter_scales,
            "K" : opts.K,
            "L" : opts.L, 
            "alpha" : opts.alpha 
          }
          np.save(join(opts.out_dir, "hyper_params.npy"), hyper_params)
    
if __name__ == '__main__':
  main()
