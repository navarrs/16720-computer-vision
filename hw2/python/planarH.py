import numpy as np
import scipy.linalg
import cv2

def computeH(x1, x2):
  def createA():
    n = x1.shape[0]
    A = np.zeros((2*n, 9), dtype=np.float)
    r = 0
    for i in range(n):
      # print(f"x1 {x1[i]} x2 {x2[i]}\n")
      x, y   = x2[i, 0], x2[i, 1]
      x_, y_ = x1[i, 0], x1[i, 1] 
      
      # x2	y2	1	0	0 0 -x2x1 -y2x1 - x1
      A[r, :] = [x, y, 1, 0, 0, 0, -x * x_, -y * x_, -x_] 
      r += 1
      # 0 0 0 x2	y2	1	0	0 0 -x2y1 -y2 y1 -y1
      A[r, :] = [0, 0, 0, x, y, 1, -x * y_, -y * y_, -y_]
      r += 1
      # print(f"A:\n{A}")
    return A
  
  # Q2.2.1
  # Compute the homography between two sets of points
  assert x1.shape == x2.shape, "Sets of points have different sizes"
  
  A = createA()
  A_ = np.matmul(np.transpose(A), A)
  _, U = np.linalg.eig(A_)
  H2to1 = U[:, -1].reshape((3, 3))
  # H2to1 = V_T[-1, :].reshape((3, 3))
  return H2to1


def computeH_norm(x1, x2):
  # Q2.2.2
  print(f"Input x1:\n{x1}")
  print(f"Input x2:\n{x2}")
  n = x1.shape[0]
  
  # Compute the centroid of the points
  x1_c = np.mean(x1, axis=0, dtype=np.float)
  print(f"Centroid x1: {x1_c}")
  x2_c = np.mean(x2, axis=0, dtype=np.float)
  print(f"Centroid x2: {x2_c}")
  
  # Shift the origin of the points to the centroid
  x1_s = x1 - x1_c
  print(f"Shifted x1:\n{x1_s}")
  x2_s = x2 - x2_c
  print(f"Shifted x2:\n{x2_s}")
  
  # Normalize the points so that the largest distance from the origin is equal 
  # to sqrt(2)  
  x1_sc = np.sqrt((1/(2*n)) * np.sum(x1_s**2))
  x1_sc = 1/x1_sc
  x1_n = x1_sc * x1_s
  print(f"With scale: {x1_sc} normalized x1:\n{x1_n}")
  
  x2_sc = np.sqrt((1/(2*n)) * np.sum(x2_s**2))
  x2_sc = 1/x2_sc
  x2_n = x2_sc * x2_s
  print(f"With scale: {x2_sc} normalized x2:\n{x2_n}")
  
  # Similarity Transform 1
  T1 = np.matrix([[x1_sc, 0, -x1_c[0]*x1_sc], 
                  [0, x1_sc, -x1_c[1]*x1_sc], 
                  [0, 0, 1]], dtype=np.float)
  print(f"T1\n{T1}")
  
  # Similarity Transform 2
  T2 = np.matrix([[x2_sc, 0, -x2_c[0]*x2_sc], 
                  [0, x2_sc, -x2_c[1]*x2_sc], 
                  [0, 0, 1]], dtype=np.float)
  print(f"T2\n{T2}")
  
  # Compute Homography
  H2to1 = computeH(x1_n, x2_n)
  print(f"Homography:\n{H2to1}")
  
  # Denormalization
  x2 = np.c_[x2, np.ones((n, 1), dtype=np.float)]
  # x1_dn = np.matmul(np.matmul(np.matmul(np.linalg.inv(T1), H2to1), T2), np.transpose(x2)).reshape(n, 3)
  x1_dn = (np.linalg.inv(T1) * H2to1 * T2 * np.transpose(x2)).reshape(n, 3)
  print(f"Denormalization:\n{x1_dn}")
  return H2to1


# def computeH_ransac(locs1, locs2, opts):
#     # Q2.2.3
#     # Compute the best fitting homography given a list of matching points
#     max_iters = opts.max_iters  # the number of iterations to run RANSAC for
#     # the tolerance value for considering a point to be an inlier
#     inlier_tol = opts.inlier_tol

#     return bestH2to1, inliers


# def compositeH(H2to1, template, img):

#     # Create a composite image after warping the template image on top
#     # of the image using the homography

#     # Note that the homography we compute is from the image to the template;
#     # x_template = H2to1*x_photo
#     # For warping the template to the image, we need to invert it.

#     # Create mask of same size as template

#     # Warp mask by appropriate homography

#     # Warp template by appropriate homography

#     # Use mask to combine the warped template and the image

#     return composite_img
