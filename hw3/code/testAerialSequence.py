import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
import os
# write your script here, we recommend the above libraries for making your animation
from SubtractDominantMotion import SubtractDominantMotion as SDM
from LucasKanadeAffine import LucasKanadeAffine as LKA

OUT_DIR = "../out/q2-air-ica"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
else:
    import shutil
    shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, 
                    help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, 
                    help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, 
                    help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')
capture = [0, 1, 30, 60, 90, 120]

# M = LKA(seq[:, :, 0], seq[:, :, 1], threshold, num_iters)

for i in range(seq.shape[2]-1):
    
    if i not in capture:
        continue
    
    mask = SDM(seq[:, :, i], seq[:, :, i+1], threshold, num_iters, tolerance)
    
    if i in capture:
        plt.imshow(seq[:, :, i], cmap='gray')
        
        scatt = np.where(mask == True)
        plt.scatter(scatt[1], scatt[0], s=2, c='b', alpha=0.5)
        
        plt.savefig(OUT_DIR + f"/aerialseq_{i}.png")

plt.close()