import warnings

import cv2
import numpy as np


from scipy.ndimage import measurements
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import affine_transform, map_coordinates
from scipy.ndimage.morphology import (distance_transform_cdt,
                                      distance_transform_edt)
from skimage import morphology as morph

from tensorpack.dataflow.imgaug import ImageAugmentor
from tensorpack.utils.utils import get_rng

from misc.utils import cropping_center, bounding_box, get_inst_centroid

####
class GenInstance(ImageAugmentor):
    def __init__(self, crop_shape=None):
        super(GenInstance, self).__init__()
        self.crop_shape = crop_shape
    
    def reset_state(self):
        self.rng = get_rng(self)

    def _fix_mirror_padding(self, ann):
        """
        Deal with duplicated instances due to mirroring in interpolation
        during shape augmentation (scale, rotation etc.)
        """
        current_max_id = np.amax(ann)
        inst_list = list(np.unique(ann))
        inst_list.remove(0) # 0 is background
        for inst_id in inst_list:
            inst_map = np.array(ann == inst_id, np.uint8)
            remapped_ids = measurements.label(inst_map)[0]
            remapped_ids[remapped_ids > 1] += int(current_max_id)
            ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
            current_max_id = np.amax(ann)
        return ann
####


####
class GenInstanceOrd(GenInstance):   
    """
        Generate an ordinal distance map based on the instance map
        First, the euclidead distance map will be calculated. Then, the ordinal map is generated based on the euclidean distance map
    """

    def _augment(self, img, _):
        img = np.copy(img)
        orig_ann = img[...,0].astype(np.int32) # instance ID map
        fixed_ann = self._fix_mirror_padding(orig_ann)
        # re-cropping with fixed instance id map
        crop_ann = cropping_center(fixed_ann, self.crop_shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            crop_ann = morph.remove_small_objects(crop_ann, min_size=7)
        
        inst_list = list(np.unique(crop_ann))
        # inst_centroid = get_inst_centroid(crop_ann)
        inst_list.remove(0)
        mask = np.zeros_like(fixed_ann, dtype=np.float32)
        for inst_id in inst_list:
            inst_id_map = np.copy(np.array(fixed_ann == inst_id, dtype=np.uint8))
            M = cv2.moments(inst_id_map)
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            inst_id_map[fixed_ann != inst_id] = 2
            inst_id_map[int(cy), int(cx)] = 0
            inst_id_map = distance_transform_edt(inst_id_map)
            inst_id_map[fixed_ann != inst_id] = 0
            max_val = np.max(inst_id_map)
            inst_id_map = inst_id_map / max_val
            mask[fixed_ann == inst_id] = inst_id_map[fixed_ann == inst_id]
        
        def gen_ord(euc_map):
            lut_gt = [1, 0.83, 0.68, 0.54, 0.41, 0.29, 0.19, 0.09, 0]
            zeros = np.zeros_like(euc_map)
            ones = np.ones_like(euc_map)
            decoded_label = np.full(euc_map.shape, 0, dtype=np.float32)
            for k in range(8):
                if k != 7:
                    decoded_label += np.where((euc_map <= lut_gt[k]) & (euc_map > lut_gt[k+1]), ones * (k + 1), zeros)
                else:
                    decoded_label += np.where((euc_map <= lut_gt[k]) & (euc_map >= lut_gt[k+1]), ones * (k + 1), zeros)
            return decoded_label
        
        ord_map = gen_ord(mask)
        img = img.astype('float32')
        img = np.dstack([img, ord_map])

        return img
####

class GaussianBlur(ImageAugmentor):
    """ Gaussian blur the image with random window size"""
    def __init__(self, max_size=3):
        """
        Args:
            max_size (int): max possible Gaussian window size would be 2 * max_size + 1
        """
        super(GaussianBlur, self).__init__()
        self.max_size = max_size

    def _get_augment_params(self, img):
        sx, sy = self.rng.randint(1, self.max_size, size=(2,))
        sx = sx * 2 + 1
        sy = sy * 2 + 1
        return sx, sy

    def _augment(self, img, s):
        return np.reshape(cv2.GaussianBlur(img, s, sigmaX=0, sigmaY=0,
                                           borderType=cv2.BORDER_REPLICATE), img.shape)

####
class BinarizeLabel(ImageAugmentor):
    """ Convert labels to binary maps"""
    def __init__(self):
        super(BinarizeLabel, self).__init__()

    def _get_augment_params(self, img):
        return None

    def _augment(self, img, s):
        img = np.copy(img)
        arr = img[...,0]
        arr[arr > 0] = 1
        return img

####
class MedianBlur(ImageAugmentor):
    """ Median blur the image with random window size"""
    def __init__(self, max_size=3):
        """
        Args:
            max_size (int): max possible window size 
                            would be 2 * max_size + 1
        """
        super(MedianBlur, self).__init__()
        self.max_size = max_size

    def _get_augment_params(self, img):
        s = self.rng.randint(1, self.max_size)
        s = s * 2 + 1
        return s

    def _augment(self, img, ksize):
        return cv2.medianBlur(img, ksize)

