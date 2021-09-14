
import glob
import os

import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (binary_dilation, binary_fill_holes,
                                      distance_transform_cdt,
                                      distance_transform_edt)
from skimage.morphology import remove_small_objects, watershed
import matplotlib.pyplot as plt

import postproc.post_sonnet

from config import Config

from misc.viz_utils import visualize_instances
from misc.utils import get_inst_centroid
from metrics.stats_utils import remap_label

##########

## ! WARNING: 
## check the prediction channels, wrong ordering will break the code !
## the prediction channels ordering should match the ones produced in augs.py

cfg = Config()
 

pred_dir = cfg.inf_output_dir
proc_dir = pred_dir + '/_proc'

file_list = glob.glob('%s/*.mat' % (pred_dir))
file_list.sort() # ensure same order

if not os.path.isdir(proc_dir):
    os.makedirs(proc_dir)

for filename in file_list:
    filename = os.path.basename(filename)
    basename = filename.split('.')[0]
    print(pred_dir, basename, end=' ', flush=True)

        
    pred_mat = sio.loadmat('%s/%s.mat' % (pred_dir, basename))
    pred = np.squeeze(pred_mat['result'])
    pred_ord = np.squeeze(pred_mat['result-ord'])

    if hasattr(cfg, 'type_classification') and cfg.type_classification:
        pred_inst = pred[...,cfg.nr_types:]
        pred_type = pred[...,:cfg.nr_types]

        pred_inst = np.squeeze(pred_inst)
        pred_type = np.argmax(pred_type, axis=-1)
        ###
 
    pred_inst = postproc.post_sonnet.proc_np_ord(pred_inst, pred_ord)

    # ! will be extremely slow on WSI/TMA so it's advisable to comment this out
    # * remap once so that further processing faster (metrics calculation, etc.)
    pred_inst = remap_label(pred_inst, by_size=True)


    if cfg.type_classification:                   
        #### * Get class of each instance id, stored at index id-1
        pred_id_list = list(np.unique(pred_inst))[1:] # exclude background ID
        pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
        for idx, inst_id in enumerate(pred_id_list):
            inst_type = pred_type[(pred_inst == inst_id)&(pred_ord>=5)]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            try:
                inst_type = type_list[0][0]
            except IndexError:
                inst_type = 0
            if inst_type == 0: # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    if (type_list[1][1] / type_list[0][1]) > 0.5:
                        inst_type = type_list[1][0]
                    else:
                        inst_type = type_list[0][0]
                        print('[Warn] Instance has `background` type')
                else:
                    print('[Warn] Instance has `background` type' )
            pred_inst_type[idx] = inst_type
        pred_inst_centroid = get_inst_centroid(pred_inst)

        sio.savemat('%s/%s.mat' % (proc_dir, basename), 
                    {'inst_map'  :     pred_inst,
                     'inst_type' :     pred_inst_type[:, None], 
                     'inst_centroid' : pred_inst_centroid,
                     'type_map'  : pred_type
                    })

    ##
    print('FINISH')
