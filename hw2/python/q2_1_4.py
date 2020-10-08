import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts

def ablation(opts, cv_cover, cv_desk):
    sigmas = np.arange(0.1, 1.0, 0.1)
    ratios = np.arange(0.1, 1.0, 0.1)
    for s in sigmas:
        for r in ratios:
            opts.ratio = r
            opts.sigma = s
            print("Using ratio {:.2f}, sigma {:.2f}".format(r, s))

            cvc = cv_cover.copy()
            cvd = cv_desk.copy()

            matches, locs1, locs2 = matchPics(cvc, cvd, opts)
            plotMatches(cv_cover, cv_desk, matches, locs1, locs2, opts)


opts = get_opts()

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

if opts.ablation:
    ablation(opts, cv_cover, cv_desk)
else:
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
    # display matched features
    plotMatches(cv_cover, cv_desk, matches, locs1, locs2, opts)
