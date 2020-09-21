import os
import multiprocessing
from os.path import join, isfile
import glob
import numpy as np
from PIL import Image
import scipy.ndimage
import scipy.signal
import skimage.color
from enum import Enum
from sklearn.cluster import KMeans
from itertools import repeat

################################################################################
# Q1.1
class Filter(Enum):
  """ 
    Enum containing the types of filters types available
  """
  GAUSSIAN = 1
  LOG = 2
  DOG_X = 3
  DOG_Y = 4

def get_response(img, ftype, scale):
  """
    Gets a filter response
    [input]
    ------
      * img: an array corresponding to one channel of an image
      * ftype: type of filter to apply to img
      * scale: sigma of the filter
    [output]
    --------
      * response of the same size as img 
  """
  if Filter.GAUSSIAN == ftype:
    return scipy.ndimage.gaussian_filter(img, scale, order=0, mode='reflect')
  elif Filter.LOG == ftype:
    return scipy.ndimage.gaussian_laplace(img, scale, mode='reflect')
  elif Filter.DOG_X == ftype:
    r1 = scipy.ndimage.gaussian_filter1d(img, scale, axis=1)
    r2 = scipy.ndimage.gaussian_filter1d(img, 2*scale, axis=1)
    return r2 - r1
  elif Filter.DOG_Y == ftype:
    r1 = scipy.ndimage.gaussian_filter1d(img, scale, axis=0)
    r2 = scipy.ndimage.gaussian_filter1d(img, 2*scale, axis=0)
    return r2 - r1

def extract_filter_responses(opts, img):
  '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
  '''
  # Check if the image is floating point type and in range [0, 1]
  if img.dtype != np.float32:
      if img.dtype.kind == 'u':
          img = img.astype(np.float32) / np.iinfo(img.dtype).max
      else:
          print("Unsupported conversion")
          return 0

  # Check if image is grayscale, if so, convert it into three channel image
  if len(img.shape) == 2:
    img = np.array([img, img, img])
    img = np.moveaxis(img, 0, 2)

  # Convert image from RGB to Lab
  img = skimage.color.rgb2lab(img)
  
  filter_scales = opts.filter_scales
  filter_responses = np.ones(
    (img.shape[0], img.shape[1], len(Filter) * len(filter_scales) * img.shape[2]))
  
  # Compute responses
  i = 0
  for fscale in filter_scales:
    for ftype in Filter:
      for c in range(img.shape[2]):
        filter_responses[:, :, i] = get_response(img[:, :, c], ftype, fscale)
        i += 1
        
  return filter_responses

################################################################################
# Q1.2
def compute_dictionary_one_image(opts, img_path):
  '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    [input]
    * opts     : options
    * img_path : path to an input image
    [saves]
    * image_response: filter response of image
  '''
  # Load image 
  img = Image.open(join(opts.data_dir, img_path))
  img = np.array(img).astype(np.float32)/255
  filter_responses = extract_filter_responses(opts, img)
  
  # Sample pixel locations 
  sub_response = np.zeros((opts.alpha, filter_responses.shape[2]))
  for a in range(opts.alpha):
    # Sample a location in the image 
    r = np.random.randint(low=0, high=img.shape[0])
    c = np.random.randint(low=0, high=img.shape[1])
    sub_response[a] = filter_responses[r, c, :]
  np.save(join(opts.data_dir, img_path.split(".jpg")[0])+".npy", sub_response)
  
def compute_dictionary(opts, n_worker=1):
  '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
  '''
  data_dir = opts.data_dir
  feat_dir = opts.feat_dir
  out_dir = opts.out_dir
  K = opts.K
  alpha = opts.alpha
  train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
  
  # Sequential 
  # for tfile in train_files:
  #   compute_dictionary_one_image(opts, tfile)
  
  # Parallel
  pool = multiprocessing.Pool(n_worker)
  for tfile in train_files:
    pool.apply_async(compute_dictionary_one_image, [opts, tfile])
  pool.close()
  pool.join()
  
  # Collect all responses back 
  responses = np.zeros((alpha * len(train_files), 4 * 3 * len(opts.filter_scales)))
  responses_files = glob.glob(data_dir + "/**/*.npy", recursive=True)
  i = 0
  for rfile in responses_files:
    with open(rfile, "rb") as f:
      responses[i:i+alpha, :] = np.load(f)
    i += alpha
    
  # Run k-means 
  kmeans = KMeans(n_clusters=K).fit(responses)
  dictionary = kmeans.cluster_centers_
  np.save(join(out_dir, 'dictionary.npy'), dictionary)

################################################################################
# Q1.3
def get_visual_words(opts, img, dictionary):
  '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
  '''
  wordmap = np.zeros((img.shape[0], img.shape[1]), dtype=float)
  response = extract_filter_responses(opts, img)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      dist = scipy.spatial.distance.cdist(response[i, j].reshape(1, response.shape[2]), 
                                          dictionary, metric='euclidean')
      wordmap[i, j] = np.argmin(dist)
  return wordmap