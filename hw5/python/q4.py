import numpy as np

import skimage
import skimage.measure as Me
import skimage.color as C
import skimage.restoration as R
import skimage.filters as F
import skimage.morphology as M
import skimage.segmentation as S
# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################
    # blur
    sigma_est = R.estimate_sigma(image, average_sigmas=True, multichannel=True)
    patch_kw = dict(patch_size=5,  # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)
    image = R.denoise_nl_means(image, h=1.15*sigma_est, fast_mode=True, **patch_kw)
    # image = R.denoise_bilateral(image, sigma_color=sigma_est, sigma_spatial=15, multichannel=True)
    # image = R.denoise_tv_chambolle(image, weight=0.1, multichannel=True)
    image = C.rgb2grey(image)
    
    # import matplotlib.pyplot as plt
    # fig, ax = F.try_all_threshold(image, figsize=(10, 8), verbose=False)
    # plt.show()

    threshold = F.threshold_otsu(image)
    bw = image < threshold
    bw = M.binary_dilation(bw, M.square(5))
    bw = M.binary_dilation(bw, M.square(5))
    bw = M.closing(bw, M.square(5))
    cleared = S.clear_border(bw)
    labels = Me.label(cleared, connectivity=2)
    
    for region in Me.regionprops(labels):
        if region.area > 100:
            bboxes.append(region.bbox)
    
    return bboxes, bw