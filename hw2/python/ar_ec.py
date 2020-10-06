# @author Ingrid Navarro
#
# @brief q4.1
# ------------------------------------------------------------------------------
import numpy as np
import cv2
from time import time

# Includes
# ------------------------------------------------------------------------------
from opts import get_opts
import os

from planarH import *
import matplotlib.pyplot as plt


# Global parameters
# ------------------------------------------------------------------------------
# ORB feature descriptor
orb = cv2.ORB_create()
# BF Matcher
bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

opts = get_opts()

# Functions
# ------------------------------------------------------------------------------


def detect_features(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, desc = orb.detectAndCompute(img, None)
    return kp, desc


def match(desc1, kp1, desc2, kp2, dist_thrsh=0.7):
    matches = bfm.match(desc1, desc2)
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,
    #                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(img3),plt.show()
    locs1 = []
    locs2 = []
    for i, m in enumerate(matches):
        if i < len(matches) - 1 and m.distance < dist_thrsh * matches[i+1].distance:
            # print(f"Found match: {i} with distance {m.distance}")
            locs1.append([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]])
            locs2.append([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])
    return np.asarray(locs1), np.asarray(locs2)

def warp(H, template, img):
  mask = np.ones_like(template)
  mask_warp = cv2.warpPerspective(mask, H, (img.shape[1], img.shape[0]))
  templ_warp = cv2.warpPerspective(template, H, (img.shape[1], img.shape[0]))
  composite_img = (1-mask_warp) * img + templ_warp
  return composite_img

def main():
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    h1, w1, _ = cv_cover.shape
    cap_bk = cv2.VideoCapture('../data/book.mov')

    if not os.path.exists("../data/ar_prep.avi"):
        cap_ar = cv2.VideoCapture('../data/ar_source.mov')

        print(f"Preprocessing video...")
        cap_out = cv2.VideoWriter('../data/ar_prep.avi',
                                  cv2.VideoWriter_fourcc(*'XVID'), 24, (w1, h1))
        c = 0
        fmin, fmax = 100, 1000
        while cap_ar.isOpened():
            ret, frame = cap_ar.read()
            c += 1
            if ret and c > fmin:
                h2, w2, _ = frame.shape
                frame = frame[50:-50, int((w2-w1)/2):int((w2+w1)/2)]
                frame = cv2.resize(frame, dsize=(w1, h1))
                cap_out.write(frame)
            if c == fmax:
              break
        cap_out.release()
        print("Done.")
    else:
        cap_ar = cv2.VideoCapture('../data/ar_prep.avi')

    # Feature matching
    kp1, desc1 = detect_features(cv_cover)

    start = time()
    frames = 0
    opts.max_iters = 100
    while cap_bk.isOpened() and cap_ar.isOpened():
        frames += 1
        # get video frames
        ret1, bk_frame = cap_bk.read()
        ret2, ar_frame = cap_ar.read()

        if ret1 and ret2:
            h2, w2, _ = ar_frame.shape
            kp2, desc2 = detect_features(bk_frame)
            locs1, locs2 = match(desc1, kp1, desc2, kp2)

            if len(locs1) >= 4:
                # H, inliers = computeH_ransac(locs1, locs2, opts)
                H, m = cv2.findHomography(locs1, locs2, cv2.RANSAC)
                composite_img = warp(H, ar_frame, bk_frame)
                cv2.imshow('AR', composite_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    end = time()
    print(f"FPS {frames / (end - start)}")
    cap_bk.release()
    cap_ar.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()