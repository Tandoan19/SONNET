import cv2
import numpy as np
from numpy.random import default_rng
from scipy.ndimage import filters, measurements, find_objects
from scipy.ndimage.morphology import (binary_dilation, binary_fill_holes,
                                      distance_transform_cdt,
                                      distance_transform_edt)
from skimage.morphology import remove_small_objects, binary_erosion
import tensorflow as tf
from matplotlib import cm
from skimage import measure
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from collections import OrderedDict

def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    Arguments:
      - value: input tensor, NHWC ('channels_last')
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```

    Returns a 3D tensor of shape [height, width, 3], uint8.
    """

    # normalize
    if vmin is None:
        vmin = tf.reduce_min(value, axis=[1, 2])
        vmin = tf.reshape(vmin, [-1, 1, 1])
    if vmax is None:
        vmax = tf.reduce_max(value, axis=[1, 2])
        vmax = tf.reshape(vmax, [-1, 1, 1])
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # squeeze last dim if it exists
    # NOTE: will throw error if use get_shape()
    # value = tf.squeeze(value)

    # quantize
    value = tf.round(value * 255)
    indices = tf.cast(value, np.int32)

    # gather
    colormap = cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = colormap(np.arange(256))[:, :3]
    colors = tf.constant(colors, dtype=tf.float32)
    value = tf.gather(colors, indices)
    value = tf.cast(value * 255, tf.uint8)
    return value

def proc_np_ord(pred, pred_ord):
    """
    Process Nuclei Prediction with The ordinal map

    Args:
        pred: prediction output (NP branch) 
        pred_ord: ordinal prediction output (ordinal branch) 
    """
    
    blb_raw = pred

    pred_ord = np.squeeze(pred_ord)
    distance = -pred_ord
    marker = np.copy(pred_ord)
    marker[marker <= 4] = 0
    marker[marker > 4] = 1
    marker = binary_dilation(marker, iterations=1)
    # marker = binary_erosion(marker)
    # marker = binary_erosion(marker)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    # marker = cv2.morphologyEx(np.float32(marker), cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    # Processing
    blb = np.copy(blb_raw)
    blb[blb >= 0.5] = 1
    blb[blb < 0.5] = 0

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    markers = marker * blb

    proced_pred = watershed(distance, markers, mask=blb)  

    return proced_pred