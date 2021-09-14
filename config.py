import importlib

import cv2
import numpy as np
import tensorflow as tf
from tensorpack import imgaug

from loader.augs import (BinarizeLabel, GaussianBlur,
                         GenInstanceOrd, MedianBlur)

#### 
class Config(object):
    def __init__(self, ):

        self.seed = 9
        self.model_type = 'sonnet'
        self.data_type = 'consep'

        self.type_classification = True
        self.nr_types = 5
        self.nr_classes = 2 # Nuclei Pixels vs Background

        # define your nuclei type name here, please ensure it contains
        # same the amount as defined in `self.nr_types` . ID 0 is preserved
        # for background so please don't use it as ID
        if self.data_type == 'consep':
            self.nuclei_type_dict = {
                'Miscellaneous': 1, # ! Please ensure the matching ID is unique
                'Inflammatory' : 2,
                'Epithelial'   : 3,
                'Spindle'      : 4,
            }
        elif self.data_type == 'monusac':
            self.nuclei_type_dict ={
                'Epithelial' : 1,
                'Lymphocyte' : 2,
                'Macrophages': 3,
                'Neutrophil' : 4
            }
        elif self.data_type == 'pannuke':
            self.nuclei_type_dict ={
                'Neoplastic' : 1,
                'Inflammatory' : 2,
                'Connective': 3,
                'Dead' : 4,
                'Non-Neoplastic Epithelial' : 5
            }
        else:
            self.nuclei_type_dict ={
                'Other' : 1,
                'Lymphocyte' : 2,
                'Epithelial' : 3
            }
        assert len(self.nuclei_type_dict.values()) == self.nr_types - 1

        #### Dynamically setting the config file into variable
        config_file = importlib.import_module('opt.hyperconfig') # np_hv, np_dist
        config_dict = config_file.__getattribute__(self.model_type)

        for variable, value in config_dict.items():
            self.__setattr__(variable, value)
        #### Training data

        # patches are stored as numpy arrays with N channels
        # ordering as [Image][Nuclei Pixels][Nuclei Type][Additional Map]
        # Ex: with type_classification=True
        #     HoVer-Net: RGB - Nuclei Pixels - Type Map - Horizontal and Vertical Map
        # Ex: with type_classification=False
        #     Dist     : RGB - Nuclei Pixels - Distance Map
        if self.data_type != 'pannuke':
            data_code_dict = {
                'sonnet'            : '540x540_76x76',
            }
        else:
            data_code_dict = {
                'sonnet'            : '270x270_76x76',
            }

        self.data_ext = '.npy' 
        # list of directories containing validation patches. 
        # For both train and valid directories, a comma separated list of directories can be used
        self.train_dir = ['/media/tandoan/data2/CoNSeP/Train/%s/'  % data_code_dict[self.model_type]]
        # Used train_test_split alr
        self.valid_dir = ['/home/tandoan/work/PanNuke/Valid/%s' % data_code_dict[self.model_type]]

        # number of processes for parallel processing input
        self.nr_procs_train = 8
        self.nr_procs_valid = 4 

        self.input_norm  = True # normalize RGB to 0-1 range

        ####
        exp_id = 'v1.0'
        model_id = '%s' % self.model_type
        self.model_name = '%s/%s' % (exp_id, model_id)
        # loading chkpts in tensorflow, the path must not contain extra '/' 
        self.log_path = '/media/tandoan/data2/logs/logs_test'
        self.save_dir = '%s/%s' % (self.log_path, self.model_name) # log file destination

        #### Info for running inferencee
        self.inf_auto_find_chkpt = False
        # path to checkpoints will be used for inference, replace accordingly
        self.inf_model_path  = '/media/tandoan/data2/logs/logs_focalnet_noguide_consep/v1.0/focalnet/02/model-39650.index'

        # output will have channel ordering as [Nuclei Type][Nuclei Pixels][Additional]
        # where [Nuclei Type] will be used for getting the type of each instance
        # while [Nuclei Pixels][Additional] will be used for extracting instance

        self.inf_imgs_ext = '.png'
        self.inf_data_dir = '/media/tandoan/data2/CoNSeP/Test/Images'
        self.inf_output_dir = 'output/test/'

        # for inference during evalutaion mode i.e run by infer.py
        self.eval_inf_input_tensor_names = ['images']
        
        
        
        # for inference during training mode i.e run by trainer.py
        if self.model_type == 'sonnet':
            self.train_inf_output_tensor_names = ['predmap-coded', 'truemap-coded']
            self.eval_inf_output_tensor_names  = ['predmap-coded', 'predmap-ord']


    def get_model(self, phase=1): 
        if phase!=2:
            model_constructor = importlib.import_module('model.sonnet')
            model_constructor = model_constructor.Sonnet
        else:
            model_constructor = importlib.import_module('model.sonnet_v2')
            model_constructor = model_constructor.Sonnet_phase2
        return model_constructor # NOTE return alias, not object


    # refer to https://tensorpack.readthedocs.io/modules/dataflow.imgaug.html for 
    # information on how to modify the augmentation parameters
    def get_train_augmentors(self, input_shape, output_shape, view=False):
        if self.data_type != 'pannuke':
            shape_augs = [
                imgaug.Affine(
                        shear=5, # in degree
                        scale=(0.8, 1.2),
                        rotate_max_deg=179,
                        translate_frac=(0.01, 0.01),
                        interp=cv2.INTER_NEAREST,
                        border=cv2.BORDER_CONSTANT),
                imgaug.Flip(vert=True),
                imgaug.Flip(horiz=True),
                imgaug.CenterCrop(input_shape),
            ]
        else:
            shape_augs =[
                imgaug.Flip(vert=True),
                imgaug.Flip(horiz=True),
            ]

        input_augs = [
            imgaug.RandomApplyAug(
                imgaug.RandomChooseAug(
                    [
                    GaussianBlur(),
                    MedianBlur(),
                    imgaug.GaussianNoise(),
                    ]
                ), 0.5),
            # standard color augmentation
            imgaug.RandomOrderAug(
                [imgaug.Hue((-8, 8), rgb=True), 
                imgaug.Saturation(0.2, rgb=True),
                imgaug.Brightness(26, clip=True),  
                imgaug.Contrast((0.75, 1.25), clip=True),
                ]),
            imgaug.ToUint8(),
        ]

        label_augs = []
        if self.model_type == 'sonnet':
            label_augs = [GenInstanceOrd(crop_shape=output_shape)]

        if not self.type_classification:            
            label_augs.append(BinarizeLabel())

        if not view:
            label_augs.append(imgaug.CenterCrop(output_shape))        

        return shape_augs, input_augs, label_augs

    def get_valid_augmentors(self, input_shape, output_shape, view=False):
        shape_augs = [
            imgaug.CenterCrop(input_shape),
        ]

        input_augs = None

        label_augs = []
        if self.model_type == 'sonnet':
            label_augs = [GenInstanceOrd(crop_shape=output_shape)]
        label_augs.append(BinarizeLabel())

        if not view:
            label_augs.append(imgaug.CenterCrop(output_shape))        

        return shape_augs, input_augs, label_augs
