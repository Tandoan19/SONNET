import math
import time
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.ndimage import measurements

from .utils import cropping_center
from .utils import rm_n_mkdir

#####
class PatchExtractor(object):
    """
    Extractor to generate patches with or without padding.
    Turn on debug mode to see how it is done.

    Args:
        x         : input image, should be of shape HWC
        win_size  : a tuple of (h, w)
        step_size : a tuple of (h, w)
        debug     : flag to see how it is done
    Return:
        a list of sub patches, each patch has dtype same as x

    Examples:
        >>> xtractor = PatchExtractor((450, 450), (120, 120))
        >>> img = np.full([1200, 1200, 3], 255, np.uint8)
        >>> patches = xtractor.extract(img, 'mirror')
    """
    def __init__(self, win_size, step_size, debug=False):

        self.patch_type = 'mirror'
        self.win_size  = win_size
        self.step_size = step_size
        self.debug   = debug
        self.counter = 0

    def __get_patch(self, x, ptx):
        pty = (ptx[0]+self.win_size[0],
               ptx[1]+self.win_size[1])
        win = x[ptx[0]:pty[0], 
                ptx[1]:pty[1]]
        assert win.shape[0] == self.win_size[0] and \
               win.shape[1] == self.win_size[1],    \
               '[BUG] Incorrect Patch Size {0}'.format(win.shape)
        if self.debug:
            if self.patch_type == 'mirror':
                cen = cropping_center(win, self.step_size)
                cen = cen[...,self.counter % 3]
                cen.fill(150)
            cv2.rectangle(x,ptx,pty,(255,0,0),2)  
            plt.imshow(x)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            self.counter += 1
        return win
    
    def __extract_valid(self, x):
        """
        Extracted patches without padding, only work in case win_size > step_size
        
        Note: to deal with the remaining portions which are at the boundary a.k.a
        those which do not fit when slide left->right, top->bottom), we flip 
        the sliding direction then extract 1 patch starting from right / bottom edge. 
        There will be 1 additional patch extracted at the bottom-right corner

        Args:
            x         : input image, should be of shape HWC
            win_size  : a tuple of (h, w)
            step_size : a tuple of (h, w)
        Return:
            a list of sub patches, each patch is same dtype as x
        """

        im_h = x.shape[0] 
        im_w = x.shape[1]

        def extract_infos(length, win_size, step_size):
            flag = (length - win_size) % step_size != 0
            last_step = math.floor((length - win_size) / step_size)
            last_step = (last_step + 1) * step_size
            return flag, last_step

        h_flag, h_last = extract_infos(im_h, self.win_size[0], self.step_size[0])
        w_flag, w_last = extract_infos(im_w, self.win_size[1], self.step_size[1])    

        sub_patches = []
        #### Deal with valid block
        for row in range(0, h_last, self.step_size[0]):
            for col in range(0, w_last, self.step_size[1]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)  
        #### Deal with edge case
        if h_flag:
            row = im_h - self.win_size[0]
            for col in range(0, w_last, self.step_size[1]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)  
        if w_flag:
            col = im_w - self.win_size[1]
            for row in range(0, h_last, self.step_size[0]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)  
        if h_flag and w_flag:
            ptx = (im_h - self.win_size[0], im_w - self.win_size[1])
            win = self.__get_patch(x, ptx)
            sub_patches.append(win)  
        return sub_patches
    
    def __extract_mirror(self, x):
        """
        Extracted patches with mirror padding the boundary such that the 
        central region of each patch is always within the orginal (non-padded)
        image while all patches' central region cover the whole orginal image

        Args:
            x         : input image, should be of shape HWC
            win_size  : a tuple of (h, w)
            step_size : a tuple of (h, w)
        Return:
            a list of sub patches, each patch is same dtype as x
        """

        diff_h = self.win_size[0] - self.step_size[0]
        padt = diff_h // 2
        padb = diff_h - padt

        diff_w = self.win_size[1] - self.step_size[1]
        padl = diff_w // 2
        padr = diff_w - padl

        pad_type = 'constant' if self.debug else 'reflect'
        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), pad_type)
        sub_patches = self.__extract_valid(x)
        return sub_patches

    def extract(self, x, patch_type):
        patch_type = patch_type.lower()
        self.patch_type = patch_type
        if patch_type == 'valid':
            return self.__extract_valid(x)
        elif patch_type == 'mirror':
            return self.__extract_mirror(x)
        else:
            assert False, 'Unknown Patch Type [%s]' % patch_type
        return

class Padding_image(object):
    """
        Padding Images to reach the minimum size using `mirror` method. Use for Monusac dataset. 
        For HoverNet, win_size is 540x540
    """
    def __init__(self, win_size):
        self.win_size = win_size
    
    def pad(self, img_dir, ann_dir, save_img_dir, save_ann_dir, pad_type='reflect'):
        file_list = glob.glob(img_dir + '/*.tif')
        file_list.sort()
        rm_n_mkdir(save_img_dir)
        rm_n_mkdir(save_ann_dir)
        for filename in file_list:
            print(filename)
            filename = os.path.basename(filename)
            basename = filename.split('.')[0]
            img = cv2.imread(img_dir + '/' + basename + '.tif')
            ann = np.load(ann_dir + '/' + basename + '.npy')
            padt, padb, padl, padr = 0, 0, 0, 0
            if img.shape[0] < 540:
                diff_h = self.win_size[0] - img.shape[0]
                padt = diff_h // 2
                padb = diff_h - padt
            if img.shape[1] < 540:
                diff_w = self.win_size[1] - img.shape[1]
                padl = diff_w // 2
                padr = diff_w - padl
            img = np.lib.pad(img, ((padt, padb), (padl, padr), (0,0)), pad_type)
            ann = np.lib.pad(ann, ((padt, padb), (padl, padr), (0,0)), pad_type)
            inst_map = ann[...,0]
            inst_map = self._fix_mirror_padding(inst_map)
            
            cv2.imwrite(save_img_dir + '/' + basename + '.tif', img)
            np.save(save_ann_dir + '/' + basename, ann)

         
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
            remapped_ids[remapped_ids > 1] += current_max_id
            ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
            current_max_id = np.amax(ann)
        return ann
            

            
#####

###########################################################################

if __name__ == '__main__':
    # toy example for debug
    # 355x355, 480x480
    xtractor = PatchExtractor((450, 450), (120, 120), debug=True)
    a = np.full([1200, 1200, 3], 255, np.uint8)
    xtractor.extract(a, 'mirror')
    xtractor.extract(a, 'valid')
    