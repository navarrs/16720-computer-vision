import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from sklearn.cluster import AgglomerativeClustering

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(
        skimage.io.imread(os.path.join('../images', img)))
    im1 = skimage.transform.resize(im1, (im1.shape[0], im1.shape[1]))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw, cmap='Greys')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.savefig(f"../out/q4/{img}")
    # plt.show()
    plt.close()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    bx = np.asarray(bboxes)[:, 0].reshape(len(bboxes), 1)
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=100).fit(bx)
    labels = clustering.labels_
    
    label = labels[0]
    bboxes = np.asarray(bboxes)
    for n in range(clustering.n_clusters_):
        idx = np.where(labels == label)[0]
        nl = np.max(idx)+1
        if nl < len(bboxes):
            label = labels[nl]
        # print(f"before {bboxes[idx]}")
        ind = np.argsort(bboxes[idx][:, 1])
        bboxes[idx] = bboxes[idx][ind]
        # print(f"after {bboxes[idx]}")
    
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    crops = []
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        crop = np.pad(bw[minr:maxr, minc:maxc], (30, 30))
        crop = 1-skimage.transform.resize(crop, (32, 32)).T
        crops.append(crop.flatten())
        # plt.imshow(crop)
        # plt.show()
        # plt.close()

    # load the weights
    # run the crops through your neural network and print them out
    # ##########################
    # ##### your code here #####
    # ##########################
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    h1 = forward(crops, params, name='layer1', activation=sigmoid)
    yp = forward(h1, params, name='output', activation=softmax)

    y_label = np.argmax(yp, axis=1)
    print(f"Text predictions for image {img} ")
    curr_label = labels[0]
    for i, y in enumerate(y_label):
        if curr_label != labels[i]:
            curr_label = labels[i]
            print(f" -- ", end='')
        print(f"{letters[y]} ", end='')
    print(f"\n")
