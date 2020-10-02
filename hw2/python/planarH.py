import numpy as np
import scipy.linalg
import cv2

def computeH(x1, x2):
  def computeSVD(A, full=False):
    U, s, V_T = scipy.linalg.svd(A)
    if full:
      m = A.shape[0]
      n = A.shape[1]
      S = scipy.linalg.diagsvd(s, m, n)
      return U, S, V_T
    return U, s, V_T
  
  def createA():
    A = np.zeros((2*x1.shape[0], 9), dtype=np.int)
    r = 0
    for i in range(x1.shape[0]):
      # print(f"x1 {x1[i]} x2 {x2[i]}\n")
      x_2, y_2 = x2[i, 0], x2[i, 1]
      x_1, y_1 = x1[i, 0], x1[i, 1] 
      
      # x2	y2	1	0	0 0 -x2x1 -y2x1
      A[r] = [x_2, y_2, 1, 0, 0, 0, -x_2 * x_1, -y_2 * x_1, -x_1] 
      r += 1
      # 0 0 0 x2	y2	1	0	0 0 -x2y1 -y2 y1
      A[r] = [0, 0, 0, x_2, y_2, 1, -x_2 * y_1, -y_2 * y_1, -y_1]
      r += 1
      # print(f"A:\n{A}")
    return A
  
  # Q2.2.1
  # Compute the homography between two sets of points
  assert x1.shape == x2.shape, "Sets of points have different sizes"
  
  A = createA()
  U, S, V_T = computeSVD(A, full=True)
  # print(f"--- A:\n{A}\nU:\n{U}\nS:\n{S}\nV_T:\n{V_T}\n")
  
  H2to1 = V_T[:, -1].reshape((3, 3))
  # print(V_T[:, -1], "\n", H2to1)
  return H2to1


# def computeH_norm(x1, x2):
#     # Q2.2.2
#     # Compute the centroid of the points

#     # Shift the origin of the points to the centroid

#     # Normalize the points so that the largest distance from the origin is equal to sqrt(2)

#     # Similarity transform 1

#     # Similarity transform 2

#     # Compute homography

#     # Denormalization

#     return H2to1


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
