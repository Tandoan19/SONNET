import argparse
import glob
import math
import os
from collections import deque

import cv2
import numpy as np
from scipy import io as sio
import matplotlib.pyplot as plt 
from skimage import measure
from scipy.ndimage import find_objects

from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils.sessinit import get_model_loader

from config import Config
from misc.utils import rm_n_mkdir

import json
import operator


####
def get_best_chkpts(path, metric_name, comparator='>'):
    """
    Return the best checkpoint according to some criteria.
    Note that it will only return valid path, so any checkpoint that has been
    removed wont be returned (i.e moving to next one that satisfies the criteria
    such as second best etc.)
    Args:
        path: directory contains all checkpoints, including the "stats.json" file
    """
    stat_file = path + '/02' + '/stats.json'
    ops = {
            '>': operator.gt,
            '<': operator.lt,
          }

    op_func = ops[comparator]
    with open(stat_file) as f:
        info = json.load(f)
    
    if comparator == '>':
        best_value  = -float("inf")
    else:
        best_value  = +float("inf")

    best_chkpt = None
    for epoch_stat in info:
        epoch_value = epoch_stat[metric_name]
        if op_func(epoch_value, best_value):
            chkpt_path = "%s/02/model-%d.index" % (path, epoch_stat['global_step'])
            if os.path.isfile(chkpt_path):
                selected_stat = epoch_stat
                best_value  = epoch_value
                best_chkpt = chkpt_path
    return best_chkpt, selected_stat


####
class Inferer(Config):

    def __gen_prediction(self, x, predictor):
        """
        Using 'predictor' to generate the prediction of image 'x'

        Args:
            x : input image to be segmented. It will be split into patches
                to run the prediction upon before being assembled back            
        """    
        step_size = [40, 40]
        msk_size = self.infer_mask_shape
        win_size = self.infer_input_shape



        def get_last_steps(length, step_size):
            nr_step = math.ceil((length - step_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)
        
        im_h = x.shape[0] 
        im_w = x.shape[1]

        padt_img, padb_img = 0, 0
        padl_img, padr_img = 0, 0

        # pad if image size smaller than msk_size (for monusac dataset)
        if im_h < msk_size[0]:
            diff_h_img = msk_size[0] - im_h
            padt_img = diff_h_img // 2
            padb_img = diff_h_img - padt_img
        if im_w < msk_size[1]:
            diff_w_img = msk_size[1] - im_w
            padl_img = diff_w_img // 2
            padr_img = diff_w_img - padl_img
        x_pad = np.lib.pad(x, ((padt_img, padb_img), (padl_img, padr_img), (0, 0)), 'reflect')
        im_h_pad = x_pad.shape[0]
        im_w_pad = x_pad.shape[1]
        

        last_h, nr_step_h = get_last_steps(im_h_pad, step_size[0])
        last_w, nr_step_w = get_last_steps(im_w_pad, step_size[1])
        diff_h = win_size[0] - step_size[0]
        padt = diff_h // 2
        padb = last_h + win_size[0] - im_h_pad


        diff_w = win_size[1] - step_size[1]
        padl = diff_w // 2
        padr = last_w + win_size[1] - im_w_pad


        x_pad = np.lib.pad(x_pad, ((padt, padb), (padl, padr), (0, 0)), 'reflect') 


        #### TODO: optimize this
        sub_patches = []
        # generating subpatches from orginal
        for row in range(0, last_h, step_size[0]):
            for col in range (0, last_w, step_size[1]):
                win = x_pad[row:row+win_size[0], 
                        col:col+win_size[1]]
                sub_patches.append(win)
        pred_coded = deque()
        pred_ord = deque()
        while len(sub_patches) > self.inf_batch_size:
            mini_batch  = sub_patches[:self.inf_batch_size]
            sub_patches = sub_patches[self.inf_batch_size:]
            mini_output = predictor(mini_batch)
            mini_coded = mini_output[0][:, 18:58, 18:58,:]
            mini_ord = mini_output[1][:, 18:58, 18:58, :]
            mini_coded = np.split(mini_coded, self.inf_batch_size, axis=0)
            pred_coded.extend(mini_coded)
            mini_ord = np.split(mini_ord, self.inf_batch_size, axis=0)
            pred_ord.extend(mini_ord)
        if len(sub_patches) != 0:
            mini_output = predictor(sub_patches)
            mini_coded = mini_output[0][:, 18:58, 18:58,:]
            mini_ord = mini_output[1][:, 18:58, 18:58, :]
            mini_coded = np.split(mini_coded, len(sub_patches), axis=0)
            pred_coded.extend(mini_coded)
            mini_ord = np.split(mini_ord, len(sub_patches), axis=0)
            pred_ord.extend(mini_ord)

        #### Assemble back into full image
        output_patch_shape = np.squeeze(pred_coded[0]).shape
        ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

        #### Assemble back into full image
        pred_coded = np.squeeze(np.array(pred_coded))
        pred_coded = np.reshape(pred_coded, (nr_step_h, nr_step_w) + pred_coded.shape[1:])

        pred_coded = np.transpose(pred_coded, [0, 2, 1, 3, 4]) if ch != 1 else \
                        np.transpose(pred_coded, [0, 2, 1, 3])
        pred_coded = np.reshape(pred_coded, (pred_coded.shape[0] * pred_coded.shape[1], 
                                         pred_coded.shape[2] * pred_coded.shape[3], ch))
        pred_coded = np.squeeze(pred_coded[:im_h_pad, :im_w_pad]) # just crop back to original size
        pred_coded = pred_coded[padt_img:padt_img+im_h, padl_img:padl_img+im_w]

        pred_ord = np.squeeze(np.array(pred_ord))
        pred_ord = np.reshape(pred_ord, (nr_step_h, nr_step_w) + pred_ord.shape[1:])
        pred_ord = np.transpose(pred_ord, [0, 2, 1, 3])
        pred_ord = np.reshape(pred_ord, (pred_ord.shape[0] * pred_ord.shape[1], pred_ord.shape[2] * pred_ord.shape[3]))
        pred_ord = np.squeeze(pred_ord[:im_h_pad, :im_w_pad])
        pred_ord = pred_ord[padt_img:padt_img+im_h, padl_img:padl_img+im_w]

        return pred_coded, pred_ord

    ####
    def run(self):

        if self.inf_auto_find_chkpt:
            print('-----Auto Selecting Checkpoint Basing On "%s" Through "%s" Comparison' % \
                        (self.inf_auto_metric, self.inf_auto_comparator))
            model_path, stat = get_best_chkpts(self.save_dir, self.inf_auto_metric, self.inf_auto_comparator)
            print('Selecting: %s' % model_path)
            print('Having Following Statistics:')
            for key, value in stat.items():
                print('\t%s: %s' % (key, value))
        else:
            model_path = self.inf_model_path

        model_constructor = self.get_model()
        pred_config = PredictConfig(
            model        = model_constructor(),
            session_init = get_model_loader(model_path),
            input_names  = self.eval_inf_input_tensor_names,
            output_names = self.eval_inf_output_tensor_names)
        predictor = OfflinePredictor(pred_config)

        save_dir = self.inf_output_dir
        file_list = glob.glob('%s/*%s' % (self.inf_data_dir, self.inf_imgs_ext))
        file_list.sort() # ensure same order

        rm_n_mkdir(save_dir)       
        for filename in file_list:
            filename = os.path.basename(filename)
            basename = filename.split('.')[0]
            print(self.inf_data_dir, basename, end=' ', flush=True)

            ##
            if self.data_type != 'pannuke':
                img = cv2.imread(self.inf_data_dir + '/' + filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.load(self.inf_data_dir + '/' + filename)
            ##
            pred_coded, pred_ord = self.__gen_prediction(img, predictor)
            sio.savemat('%s/%s.mat' % (save_dir, basename), {'result':[pred_coded], 'result-ord':[pred_ord]})
            print('FINISH')




####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    args = parser.parse_args()
        
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    inferer = Inferer()
    inferer.run()
    