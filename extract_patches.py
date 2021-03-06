
import glob
import os

import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import measurements
import scipy.io as sio

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from config import Config

###########################################################################
if __name__ == '__main__':
    
    cfg = Config()

    extract_type = 'mirror' # 'valid' for fcn8 segnet etc.
                            # 'mirror' for u-net etc.
    # check the patch_extractor.py 'main' to see the different

    # orignal size (win size) - input size - output size (step size)
    # 540x540 - 270x270 - 76x76 sonnet
    step_size = [76, 76] # should match self.train_mask_shape (config.py) 
    win_size  = [540, 540] # should be at least twice time larger than 
                           # self.train_base_shape (config.py) to reduce 
                           # the padding effect during augmentation

    xtractor = PatchExtractor(win_size, step_size)

    ### Paths to data - these need to be modified according to where the original data is stored
    img_ext = '.tif'
    img_dir = '/media/tandoan/Data/data/Monusac/Test(split)/Images'
    ann_dir = '/media/tandoan/Data/data/Monusac/Test(split)/Labels(mat)'  
    ####
    out_dir = "/media/tandoan/Data/data/Monusac/%dx%d_%dx%d" % \
                        (win_size[0], win_size[1], step_size[0], step_size[1])

    file_list = glob.glob('%s/*%s' % (img_dir, img_ext))
    file_list.sort() 

    rm_n_mkdir(out_dir)
    for filename in file_list:
        filename = os.path.basename(filename)
        basename = filename.split('.')[0]
        print(filename)

        img = cv2.imread(img_dir + '/' + basename + img_ext)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
        if cfg.type_classification:
            # assumes that ann is HxWx2 (nuclei class labels are available at index 1 of C) 
            ann = sio.loadmat(ann_dir + '/' + basename + '.mat')
            ann_inst = ann['inst_map']
            ann_type = ann['type_map']
            
            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            if cfg.data_type == 'glysac':
                ann_type[(ann_type == 1) | (ann_type == 2) | (ann_type == 9) | (ann_type == 10)] = 1
                ann_type[(ann_type == 4) | (ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 2
                ann_type[(ann_type == 8) | (ann_type == 3)] = 3
            elif cfg.data_type == 'consep':
                ann_type[(ann_type == 3) | (ann_type == 4)] = 3
                ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4
            assert np.max(ann_type) <= cfg.nr_types-1, \
                            "Only %d types of nuclei are defined for training"\
                            "but there are %d types found in the input image." % (cfg.nr_types, np.max(ann_type)) 

            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype('int32')             

        img = np.concatenate([img, ann], axis=-1)
    
        sub_patches = xtractor.extract(img, extract_type)
        for idx, patch in enumerate(sub_patches):  
            np.save("{0}/{1}_{2:03d}.npy".format(out_dir, basename, idx), patch)