import numpy as np
import scipy.linalg
import cv2
import random

def computeH(x1, x2):
  def createA():
    n = x1.shape[0]
    A = np.zeros((2*n, 9), dtype=np.float)  
    for i in range(n):
      x, y   = x2[i, 0], x2[i, 1]
      x_, y_ = x1[i, 0], x1[i, 1] 
      
      A[2*i, :] = [-x, -y, -1, 0, 0, 0, x_*x, x_*y, x_]
      A[2*i+1, :] = [0, 0, 0, -x, -y, -1, y_*x, y_*y, y_]
    return A
  
  # Q2.2.1
  # Compute the homography between two sets of points
  assert x1.shape[0] >= 4, "Less than 4 points provided"
  assert x1.shape == x2.shape, "Sets of points have different sizes"

  A = createA()
  _,_,V_T = np.linalg.svd(A)
  # Last col of V is last row of V_T
  H2to1 = V_T[-1, :].reshape((3, 3))
  return H2to1


def computeH_norm(x1, x2):
  # Q2.2.2
  # print(f"Input x1:\n{x1}")
  # print(f"Input x2:\n{x2}")
  n = x1.shape[0]
  
  # Compute the centroid of the points
  x1_c = np.mean(x1, axis=0, dtype=np.float)
  x2_c = np.mean(x2, axis=0, dtype=np.float)
  # print(f"Centroid x1: {x1_c}")
  # print(f"Centroid x2: {x2_c}")
  
  # Shift the origin of the points to the centroid
  x1_s = x1 - x1_c
  x2_s = x2 - x2_c
  # print(f"Shifted x1:\n{x1_s}")
  # print(f"Shifted x2:\n{x2_s}")
  
  # Normalize the points so that the largest distance from the origin is equal 
  # to sqrt(2)  
  scale_x1 = 0
  for i in range(n):
    scale_x1 += np.linalg.norm(x1_s[i])
  scale_x1 /= n
  scale_x1 = np.sqrt(2) / scale_x1
  x1_n = scale_x1 * x1_s
  # print(f"With scale: {scale_x1} normalized x2:\n{x1_n}")
  
  # TODO: fix parallel
  # x1_sc = np.max(np.sqrt(np.sum(x1_s **2, axis=1)))
  # x1_sc = np.sqrt((1/(n)) * np.sum(x1_s**2))
  # x1_sc = np.sqrt(2)/x1_sc
  # x1_sc = 1/x1_sc
  # x1_n = x1_sc * x1_s
  
  scale_x2 = 0
  for i in range(n):
    scale_x2 += np.linalg.norm(x2_s[i])
  scale_x2 /= n
  scale_x2 = np.sqrt(2) / scale_x2
  x2_n = scale_x2 * x2_s
  # print(f"With scale: {scale_x2} normalized x2:\n{x2_n}")
  
  # TODO: fix parallel
  # x2_sc = np.max(np.sqrt(np.sum(x2_s **2, axis=1)))
  # x2_sc = np.sqrt((1/(2*n)) * np.sum(x2_s**2))
  # x2_sc = np.sqrt(2)/x2_sc
  # x2_sc = 1/x2_sc
  # x2_n = x2_sc * x2_s
  
  # Similarity Transform 1
  T1 = np.matrix([[scale_x1, 0, -x1_c[0]*scale_x1], 
                  [0, scale_x1, -x1_c[1]*scale_x1], 
                  [0, 0, 1]], dtype=np.float)
  # print(f"T1\n{T1}")
  
  # Similarity Transform 2
  T2 = np.matrix([[scale_x2, 0, -x2_c[0]*scale_x2], 
                  [0, scale_x2, -x2_c[1]*scale_x2], 
                  [0, 0, 1]], dtype=np.float)
  # print(f"T2\n{T2}")
  
  # Compute Homography
  H2to1 = computeH(x1_n, x2_n)
  # print(f"Homography:\n{H2to1}\n")
  
  # Denormalization
  # H2to1 = np.matmul(np.matmul(np.linalg.inv(T1), H2to1), T2)
  H2to1 = np.linalg.inv(T1) @ H2to1 @ T2
  # H2to1 *= 1./H2to1[-1, -1]
  return H2to1


def computeH_ransac(locs1, locs2, opts):
  # Q2.2.3
  # Compute the best fitting homography given a list of matching points
  max_iters = opts.max_iters  # the number of iterations to run RANSAC for
  # the tolerance value for considering a point to be an inlier
  inlier_tol = opts.inlier_tol
  n = locs1.shape[0]
  
  max_inliers = 0
  
  for i in range(max_iters):
    inliers = np.zeros((n, 1), dtype=np.int)
    samples = random.sample(range(n), 4)
    inliers[samples] = 1
    x1 = locs1[samples]
    x2 = locs2[samples]
    
    H = computeH(x1, x2)
    # TODO: fix 
    # H = computeH_norm(x1, x2)
    
    x1_ = np.c_[locs2, np.ones((locs2.shape[0]))]
    x2_ = np.c_[locs1, np.ones((locs1.shape[0]))]
    
    x1_est = np.dot(H, x2_.T).T
    x1_est = np.divide(x1_est, x1_est[:, -1].reshape(x1_est.shape[0], 1))
    
    # print(f"x1 {x1_}\n x1_est {x1_est}") 
    err = x1_est - x1_
    err = np.sqrt(np.sum(err**2, axis=1))
    # err = np.linalg.norm(err, axis = 0, ord=2)
    
    inliers[err < inlier_tol] = 1
    
    inlier_count = np.sum(inliers)
    # print(f"Found {inlier_count} inliers")
    if inlier_count > max_inliers:
      bestH2to1 = H
      best_inliers = inliers
      max_inliers = inlier_count
  
  return bestH2to1, best_inliers


def compositeH(H2to1, template, img):
  # Create a composite image after warping the template image on top
  # of the image using the homography
  # Note that the homography we compute is from the image to the template;
  # x_template = H2to1*x_photo
  # For warping the template to the image, we need to invert it.
  
  # H2to1 = np.linalg.inv(H2to1)
   
  # Create mask of same size as template
  mask = np.ones_like(template)
  
  # Warp mask by appropriate homography
  mask_warp = cv2.warpPerspective(mask, H2to1, 
                                 (img.shape[1], img.shape[0]))
 
  # Warp template by appropriate homography
  templ_warp = cv2.warpPerspective(template, H2to1, 
                                 (img.shape[1], img.shape[0]))
  cv2.imshow('mask', templ_warp)
  cv2.waitKey(0)
  
  # Use mask to combine the warped template and the image
  composite_img = (1-mask_warp) * img + templ_warp
  
  return composite_img