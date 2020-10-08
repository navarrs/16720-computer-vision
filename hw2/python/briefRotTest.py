import numpy as np
import scipy
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts
import matplotlib.pyplot as plt

opts = get_opts()

# Q2.1.6
im = cv2.imread("../data/cv_cover.jpg")
# cv2.imshow('image', im)
# cv2.waitKey(0)

match_count = np.zeros((36, 2))

for i in range(36):
    # Rotate Image
    rot = (i+1)*10
    im_rot = scipy.ndimage.rotate(im, rot)
    # cv2.imshow(f'rot {i}', im_rot)
    # cv2.waitKey(0)
    # cv2.destroyWindow(f'rot {i}')

    # Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(im, im_rot, opts)
    
    # Plot them
    opts.name = f"rot_{rot}_m_{len(matches)}"
    # plotMatches(im, im_rot, matches, locs1, locs2, opts)
    
    # Update histogram
    print(f"Matches {len(matches)} Angle {rot}")
    match_count[i] = [rot, len(matches)]

print(match_count)

# Display histogram
plt.bar(match_count[:, 0], match_count[:, 1], width=8)
plt.xlabel('Orientation')
plt.ylabel('Matches')
plt.xticks(np.arange(10, 360, 10.0))
plt.title('Match count histogram')
plt.grid(True)
plt.show()