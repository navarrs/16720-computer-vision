import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from LucasKanade import LucasKanade as LK

OUT_DIR = "../out/q1-3_girlseq"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

#
# PARSING ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, 
                    help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, 
                    help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
    
# 
# MAIN -------------------------------------------------------------------------
seq = np.load("../data/girlseq.npy")
rect = [280., 152., 330., 318.]
capture = [1, 20, 40, 60, 80]
girl_rects = np.zeros((seq.shape[2], 4))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i in range(seq.shape[2]-1):

    girl_rects[i, ] = rect

    # Compute p
    p = LK(seq[:, :, i], seq[:, :, i+1], rect, threshold, num_iters)
    
    # Update bbox
    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]

    # Save images falling in this condition
    if i in capture:
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        r = plt.Rectangle((rect[0], rect[1]),
                          height=h, width=w, ec='r', lw=3, fill=False)

        plt.imshow(seq[:, :, i+1], cmap='gray')
        ax.add_patch(r)
        plt.pause(0.2)
        plt.draw()
        plt.savefig(OUT_DIR + f"/girlseq_{i}.png")
        r.remove()
plt.close()

with open(OUT_DIR + "/girlseqrects.npy", "wb") as f:
    np.save(f, girl_rects)


#
# SANITY CHECK -----------------------------------------------------------------
if args.visualize:
    with open(OUT_DIR + "/girlseqrects.npy", 'rb') as f:
        rects = np.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(seq.shape[2]-1):
        rect = rects[i]
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        r = plt.Rectangle((rect[0], rect[1]),
                          height=h, width=w, ec='b', lw=2, fill=False)
        ax.add_patch(r)
        plt.imshow(seq[:, :, i], cmap='gray')
        plt.pause(0.01)
        plt.draw()
        r.remove()
    plt.close()