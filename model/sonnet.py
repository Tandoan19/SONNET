import os
import math
import string
import collections

import tensorflow as tf 
import cv2

from tensorpack import *
from tensorpack.models import BatchNorm, BNReLU, Conv2D, GlobalAvgPooling, Dropout
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary, add_activation_summary
from tensorpack.tfutils.scope_utils import under_name_scope, auto_reuse_variable_scope
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.gradproc import GlobalNormClip


import sys
sys.path.append("..") # adds higher directory to python modules path.

from .utils import *


try: # HACK: import beyond current level, may need to restructure
    from config import Config
except ImportError:
    assert False, 'Fail to import config.py'

####
def upsample2x(name, x):
    """
    Nearest neighbor up-sampling
    """
    return FixedUnPooling(
                name, x, 2, unpool_mat=np.ones((2, 2), dtype='float32'),
                data_format='channels_first')

def res_blk(name, l, ch, ksize, count, split=1, strides=1):
    ch_in = l.get_shape().as_list()
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block' + str(i)):  
                x = l if i == 0 else BNReLU('preact', l)
                x = Conv2D('conv1', x, ch[0], ksize[0], activation=BNReLU)
                x = Conv2D('conv2', x, ch[1], ksize[1], split=split, 
                                strides=strides if i == 0 else 1, activation=BNReLU)
                x = Conv2D('conv3', x, ch[2], ksize[2], activation=tf.identity)
                if (strides != 1 or ch_in[1] != ch[2]) and i == 0:
                    l = Conv2D('convshortcut', l, ch[2], 1, strides=strides)
                l = l + x
        # end of each group need an extra activation
        l = BNReLU('bnlast',l)  
    return l

def dense_blk(name, l, ch, ksize, count, split=1, padding='valid'):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('blk/' + str(i)):
                x = BNReLU('preact_bna', l)
                x = Conv2D('conv1', x, ch[0], ksize[0], padding=padding, activation=BNReLU)
                x = Conv2D('conv2', x, ch[1], ksize[1], padding=padding, split=split)
                ##
                if padding == 'valid':
                    x_shape = x.get_shape().as_list()
                    l_shape = l.get_shape().as_list()
                    l = crop_op(l, (l_shape[2] - x_shape[2], 
                                    l_shape[3] - x_shape[3]))

                l = tf.concat([l, x], axis=1)
        l = BNReLU('blk_bna', l)
    return l

####
@layer_register(log_shape=True)
def resize_bilinear(i, size, align_corners=False):
    ret = tf.transpose(i, [0, 2, 3, 1])
    ret = tf.image.resize_bilinear(ret, size=[size, size], align_corners=align_corners)
    ret = tf.transpose(ret, [0, 3, 1, 2])
    return tf.identity(ret, name='output')

####
def resize_nearest_neighbor(i, size):
    ret = tf.transpose(i, (0, 2, 3, 1))
    ret = tf.image.resize_nearest_neighbor(ret, (size, size))
    ret = tf.transpose(ret, (0, 3, 1, 2))
    return ret

####
@layer_register(log_shape=True)
def DepthConv(x, out_channel, kernel_shape, padding='SAME', stride=1,
                W_init=None, activation=tf.identity):
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[1]
    assert out_channel % in_channel == 0, (out_channel, in_channel)
    channel_mult = out_channel // in_channel

    if W_init is None:
        W_init = tf.variance_scaling_initializer(scale=2.0, mode='fan_out')
        kernel_shape = [kernel_shape, kernel_shape]
        filter_shape = kernel_shape + [in_channel, channel_mult]

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    conv = tf.nn.depthwise_conv2d(x, W, [1, 1, stride, stride], padding=padding, data_format='NCHW')
    return activation(conv, name='output')


BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])

BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]


def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier"""
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier"""
    return int(math.ceil(depth_coefficient * repeats))

def mb_conv_block(inputs, block_args, activation, drop_rate=None, freeze_en=False, prefix='',):
    """Mobile Inverted Residual Bottleneck."""
    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 1
    # For naming convolutional, batchnorm layers to match pretrained weights
    num_conv = ''
    num_batch = 0
    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:    
        x = Conv2D(prefix + '/conv2d', inputs, filters, 1, padding='same', use_bias=False)
        num_conv = '_1'
        x = BatchNorm(prefix + '/tpu_batch_normalization', x)
        num_batch += 1
        x = activation(x)
    else:
        x = inputs
    # Depthwise convolution
    x = DepthConv(prefix+'/depthwise_conv2d', x, x.get_shape().as_list()[1], block_args.kernel_size, stride=block_args.strides[0])
    if num_batch != 0:
        x = BatchNorm(prefix + '/tpu_batch_normalization' + '_' + str(num_batch), x)
    else:
        x = BatchNorm(prefix + '/tpu_batch_normalization', x)
    num_batch += 1
    x = activation(x)
    
    # Squeeze and Excitation phase
    if has_se:
        num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))
        se_tensor = GlobalAvgPooling(prefix+'/se/squeeze', x, data_format='NCHW')
        target_shape = [-1, filters, 1, 1]
        se_tensor = tf.reshape(se_tensor, target_shape, name=prefix + '/se/reshape')
        
        se_tensor = Conv2D(prefix + '/se/conv2d', se_tensor, num_reduced_filters, 1, activation=activation, padding='same', use_bias=True)
        se_tensor = Conv2D(prefix + '/se/conv2d_1', se_tensor, filters, 1, activation=tf.sigmoid, padding='same', use_bias=True)
        x = tf.multiply(x, se_tensor, name=prefix + '/se/excite')
    # Output phase
    x = Conv2D(prefix + '/conv2d' + num_conv, x, block_args.output_filters, 1, padding='same', use_bias=False)
    x = tf.stop_gradient(x) if freeze_en else x
    x = BatchNorm(prefix + '/tpu_batch_normalization' + '_' + str(num_batch), x)
    if block_args.id_skip and all(s==1 for s in block_args.strides) and block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            x = Dropout(x, rate=drop_rate, noise_shape=(tf.shape(x)[0], 1, 1, 1), name=prefix + 'drop')
        x = tf.math.add(x, inputs, name=prefix + 'add')
    x = tf.stop_gradient(x) if freeze_en else x
    return x

def EfficientNet(i, width_coefficient, depth_coefficient, default_resolution, dropout_rate=0.2, drop_connect_rate=0.2,
                 depth_divisor=8, block_args=DEFAULT_BLOCKS_ARGS, freeze_en=False, model_name='efficientnet'):
    """Instantiates the EfficientNet architecture using given scaling coefficient.
    # Arguments:
        width_coefficient: float, scaling coefficient for network width
        depth_coefficient: float, scaling coefficient for network depth
        default_resolution: int, default resolution of input
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connection
        depth_divisor: int.
        block_args: A list of BlockArgs to construct block modules.
        model_name: string, model name.
    # Returns:
        EfficientNet as a backbone encoder.
    """    

    bn_axis = 1
    activation = tf.nn.swish
    with tf.variable_scope(model_name):
        with tf.variable_scope('stem'):
            # Build stem
            x = Conv2D('conv2d', i, round_filters(32, width_coefficient, depth_divisor), 3, strides=(1, 1), padding='same', use_bias=False)
            x = BatchNorm('tpu_batch_normalization', x)
            x = activation(x)

        # Build blocks
        num_blocks_total = sum(block_args.num_repeat for block_args in block_args)
        block_num = 0
        ret = []
        for idx, block_args in enumerate(block_args):
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, width_coefficient, depth_divisor),
                output_filters=round_filters(block_args.output_filters, width_coefficient, depth_divisor),
                num_repeat=round_repeats(block_args.num_repeat, depth_coefficient)
            )
            # The first block needs to take care of stride and filter size increase.
            drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
            x = mb_conv_block(x, block_args, activation=activation, drop_rate=drop_rate, freeze_en=freeze_en, prefix='blocks_{}'.format(block_num))
            if block_num==0 or block_num==2 or block_num==4 or block_num==10: 
                ret.append(x)
                x = tf.stop_gradient(x) if freeze_en else x
            block_num += 1
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])
                for bidx in range(block_args.num_repeat - 1):
                    drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                    block_prefix = 'blocks_{}'.format(block_num)
                    x = mb_conv_block(x, block_args, activation=activation, drop_rate = drop_rate, freeze_en=freeze_en, prefix=block_prefix)
                    if block_num==0 or block_num == 2 or block_num==4 or block_num==10 : 
                        ret.append(x)
                        x = tf.stop_gradient(x) if freeze_en else x
                    block_num += 1
    
        # Build head
        with tf.variable_scope('head'):
            x = Conv2D('conv2d', x, round_filters(1280, width_coefficient, depth_divisor), 1, padding='same', use_bias=False)
            x = tf.stop_gradient(x) if freeze_en else x
        x = Conv2D('conv_bot', x, 1024, 1)
        ret.append(x)
    return ret


###
def EfficientNetB0(x, freeze_en):
    return EfficientNet(x, 1.0, 1.0, 224, 0.2, freeze_en=freeze_en, model_name='efficientnet-b0')

###
def EfficientNetB1(x, freeze_en):
    return EfficientNet(x, 1.0, 1.1, 240, 0.2, freeze_en=freeze_en, model_name='efficientnet-b1')

###
def EfficientNetB2(x, freeze_en):
    return EfficientNet(x, 1.1, 1.2, 260, 0.3, freeze_en=freeze_en, model_name='efficientnet-b2')

###
def EfficientNetB3(x, freeze_en):
    return EfficientNet(x, 1.2, 1.4, 300, 0.3, freeze_en=freeze_en, model_name='efficientnet-b3')
###
def EfficientNetB4(x, freeze_en):
    return EfficientNet(x, 1.4, 1.8, 380, 0.4, freeze_en=freeze_en, model_name='efficientnet-b4')

####
def decoder(name, i):
    pad = 'valid'
    with tf.variable_scope(name):
        with tf.variable_scope('S5'):
            u5 = Conv2D('bottleneck', i[-1], 256, 1, strides=1, activation=BNReLU) 
            u5_x2 = upsample2x('deux', u5)
        with tf.variable_scope('S4'):
            u4 = Conv2D('bottleneck', i[-2], 256, 1, strides=1, activation=BNReLU)
            u4_add = tf.add_n([u4, u5_x2])
            u4 = Conv2D('conva', u4_add, 256, 5, strides=1, padding=pad) 
            with tf.variable_scope('_add_1'):
                u4_10 = BNReLU('preact', u4)
                u4_11 = Conv2D('_1', u4_10, 128, 1, padding=pad, dilation_rate=(1, 1), activation=BNReLU)
                u4_12 = Conv2D('_2', u4_11, 32, 3, padding=pad, dilation_rate=(1, 1))
                u4_12 = crop_op(u4_12, (6, 6)) 
            with tf.variable_scope('_add_2'):
                u4_20 = BNReLU('preact', u4)
                u4_21 = Conv2D('_1', u4_20, 128, 1, padding=pad, dilation_rate=(1, 1), activation=BNReLU)
                u4_22 = Conv2D('_2', u4_21, 32, 5, padding=pad, dilation_rate=(1, 1))
                u4_22 = crop_op(u4_22, (4, 4)) 
            with tf.variable_scope('_add_3'):
                u4_30 = BNReLU('preact', u4)
                u4_31 = Conv2D('_1', u4_30, 128, 1, padding=pad, dilation_rate=(1, 1), activation=BNReLU)
                u4_32 = Conv2D('_2', u4_31, 32, 3, padding=pad, dilation_rate=(2, 2))
                u4_32 = crop_op(u4_32, (4, 4)) 
            with tf.variable_scope('_add_4'):
                u4_40 = BNReLU('preact', u4)
                u4_41 = Conv2D('_1', u4_40, 128, 1, padding=pad, dilation_rate=(1, 1), activation=BNReLU)
                u4_42 = Conv2D('_2', u4_41, 32, 5, padding=pad, dilation_rate=(2, 2), activation=BNReLU) 
            u4 = crop_op(u4, (8, 8)) 
            u4_cat = tf.concat([u4, u4_12, u4_22, u4_32, u4_42], axis=1, name='_cat') 
            u4_cat = BNReLU('outcat_BNReLU', u4_cat)
            u4_cat = Conv2D('_outcat', u4_cat, 256, 3, padding=pad, strides=1, activation=BNReLU) 
            u4_x2 = upsample2x('deux', u4_cat)
        
        with tf.variable_scope('S3'):
            u3 = Conv2D('bottleneck', i[-3], 256, 1, strides=1, activation=BNReLU)
            u3 = crop_op(u3, (28, 28))
            u3_add = tf.add_n([u3, u4_x2]) 
            u3 = Conv2D('conva', u3_add, 256, 5, strides=1, padding=pad) 
            with tf.variable_scope('_add_1'):
                u3_10 = BNReLU('preact', u3)
                u3_11 = Conv2D('_1', u3_10, 128, 1, padding=pad, dilation_rate=(1, 1), activation=BNReLU)
                u3_12 = Conv2D('_2', u3_11, 32, 3, padding=pad, dilation_rate=(1, 1))
                u3_12 = crop_op(u3_12, (6, 6))
            with tf.variable_scope('_add_2'):
                u3_20 = BNReLU('preact', u3)
                u3_21 = Conv2D('_1', u3_20, 128, 1, padding=pad, dilation_rate=(1, 1), activation=BNReLU)
                u3_22 = Conv2D('_2', u3_21, 32, 5, padding=pad, dilation_rate=(1, 1))
                u3_22 = crop_op(u3_22, (4, 4))
            with tf.variable_scope('_add_3'):
                u3_30 = BNReLU('preact', u3)
                u3_31 = Conv2D('_1', u3_30, 128, 1, padding=pad, dilation_rate=(1, 1), activation=BNReLU)
                u3_32 = Conv2D('_2', u3_31, 32, 3, padding=pad, dilation_rate=(2, 2))
                u3_32 = crop_op(u3_32, (4, 4))
            with tf.variable_scope('_add_4'):
                u3_40 = BNReLU('preact', u3)
                u3_41 = Conv2D('_1', u3_40, 128, 1, padding=pad, dilation_rate=(1, 1), activation=BNReLU)
                u3_42 = Conv2D('_2', u3_41, 32, 5, padding=pad, dilation_rate=(2, 2))
            
            u3 = crop_op(u3, (8, 8))
            u3_cat = tf.concat([u3, u3_12, u3_22, u3_32, u3_42], axis=1, name='_cat')
            u3_cat = BNReLU('outcat_BNReLU', u3_cat)
            u3_cat = Conv2D('_outcat', u3_cat, 256, 3, padding=pad, strides=1, activation=BNReLU) 
            u3_x2 = upsample2x('deux', u3_cat) 
        
        with tf.variable_scope('S2'):
            u2 = Conv2D('bottleneck', i[-4], 256, 1, strides=1, activation=BNReLU)
            u2 = crop_op(u2, (83, 83))
            u2_add = tf.add_n([u2, u3_x2]) 
            u2 = Conv2D('conva', u2_add, 256, 5, strides=1, padding=pad) 
            with tf.variable_scope('_add_1'):
                u2_10 = BNReLU('preact', u2)
                u2_11 = Conv2D('_1', u2_10, 128, 1, padding=pad, dilation_rate=(1, 1), activation=BNReLU)
                u2_12 = Conv2D('_2', u2_11, 32, 3, padding=pad, dilation_rate=(1, 1))
                u2_12 = crop_op(u2_12, (6, 6))
            with tf.variable_scope('_add_2'):
                u2_20 = BNReLU('preact', u2)
                u2_21 = Conv2D('_1', u2_20, 128, 1, padding=pad, dilation_rate=(1, 1), activation=BNReLU)
                u2_22 = Conv2D('_2', u2_21, 32, 5, padding=pad, dilation_rate=(1, 1))
                u2_22 = crop_op(u2_22, (4, 4))
            with tf.variable_scope('_add_3'):
                u2_30 = BNReLU('preact', u2)
                u2_31 = Conv2D('_1', u2_30, 128, 1, padding=pad, dilation_rate=(1, 1), activation=BNReLU)
                u2_32 = Conv2D('_2', u2_31, 32, 3, padding=pad, dilation_rate=(2, 2))
                u2_32 = crop_op(u2_32, (4, 4))
            with tf.variable_scope('_add_4'):
                u2_40 = BNReLU('preact', u2)
                u2_41 = Conv2D('_1', u2_40, 128, 1, padding=pad, dilation_rate=(1, 1), activation=BNReLU)
                u2_42 = Conv2D('_2', u2_41, 32, 5, padding=pad, dilation_rate=(2, 2))

            u2 = crop_op(u2, (8, 8))
            u2_cat = tf.concat([u2, u2_12, u2_22, u2_32, u2_42], axis=1, name='_cat')
            u2_cat = BNReLU('outcat_BNReLU', u2_cat)
            u2_cat = Conv2D('_outcat', u2_cat, 256, 3, padding='same', strides=1, activation=BNReLU) 


        with tf.variable_scope('Pyramid'):
            p1 = Conv2D('p1', u2_cat, 128, 5, padding='same', activation=BNReLU)
            p1 = Conv2D('p1_1', p1, 128, 5, padding='same', activation=BNReLU)
            p2 = Conv2D('p2', u3_cat, 128, 5, padding=pad, activation=BNReLU)
            p2 = Conv2D('p2_1', p2, 128, 3, padding=pad, activation=BNReLU) 
            p2_x2 = upsample2x('deux_p2', p2)
            p3 = crop_op(u4_cat, (2, 2))
            p3 = Conv2D('p3', p3, 128, 5, padding=pad, activation=BNReLU)
            p3 = Conv2D('p3_1', p3, 128, 5, padding=pad, activation=BNReLU) 
            p3_x2 = upsample2x('deux_p3', p3)
            p3_x4 = upsample2x('quatre_p3', p3_x2)
            p4 = crop_op(u5, (4, 4))
            p4 = Conv2D('p4', p4, 128, 5, padding=pad, activation=BNReLU)
            p4 = Conv2D('p4_1', p4, 128, 5, padding=pad, activation=BNReLU)
            p4_x2 = upsample2x('deux_p4', p4)
            p4_x4 = upsample2x('quatre_p4', p4_x2)
            p4_x8 = upsample2x('huit_p4', p4_x4)
            p_cat = tf.concat([p1, p2_x2, p3_x4, p4_x8], axis=1) 
            p_cat = Conv2D('_outcat_1', p_cat, 256, 5, padding='same', activation=BNReLU)
            p_cat = Conv2D('_outcat_2', p_cat, 256, 5, padding='same', activation=BNReLU)
            p_cat_x2 = upsample2x('deux_cat', p_cat)
        
        i_5 = crop_op(i[0], (190, 190))
        i_5 = Conv2D('i_5', i_5, 256, 1, activation=BNReLU)
        p_cat = tf.add_n([i_5, p_cat_x2]) 
        p_0 = Conv2D('p_0', p_cat, 128, 5, strides=1, padding='valid')

    return p_0


class Model(ModelDesc, Config):
    def __init__(self, freeze_en=False):
        super(Model, self).__init__()
        assert tf.test.is_gpu_available()
        self.freeze_en = freeze_en
        self.data_format = 'NCHW'
        
    
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None] + self.train_input_shape + [3], 'images'),
                InputDesc(tf.float32, [None] + self.train_mask_shape  + [None], 'truemap-coded')]
    
    # for node to receive manual info such as learning rate.
    def add_manual_variable(self, name, init_value, summary=True):
        var = tf.get_variable(name, initializer=init_value, trainable=False)
        if summary:
            tf.summary.scalar(name + '-summary', var)
        return 
    
    def optimizer(self):
        with tf.variable_scope("", reuse=True):
            lr = tf.get_variable('learning_rate')
        opt = self.train_optimizer(learning_rate=lr)
        return optimizer.apply_grad_processors(opt, [GlobalNormClip(1.)]) 


class Sonnet(Model):

    def build_graph(self, inputs, truemap_coded):
        images = inputs
        orig_imgs = images



        if hasattr(self, 'type_classification') and self.type_classification:
            true_type = truemap_coded[..., 1]
            true_type = tf.cast(true_type, tf.int32)
            true_type = tf.identity(true_type, name='truemap-type')
            one_type = tf.one_hot(true_type, self.nr_types, axis=-1)
            true_type = tf.expand_dims(true_type, axis=-1)

            true_np = tf.cast(true_type > 0, tf.int32)
            true_np = tf.identity(true_np, name='truemap-np')
            one_np = tf.one_hot(tf.squeeze(true_np, axis=-1), 2, axis=-1)


        else:
            true_np = truemap_coded[..., 0]
            true_np = tf.cast(true_np, tf.int32)
            one_np = tf.one_hot(true_np, 2, axis=-1)
            true_np = tf.expand_dims(true_np, axis=-1)
            true_np = tf.identity(true_np, name='truemap-np')
        

        true_ord = truemap_coded[...,-1]
        true_ord = tf.expand_dims(true_ord, axis=-1)
        true_ord = tf.identity(true_ord, name='true-ord') 

        ####
        with argscope(Conv2D, activation=tf.identity, use_bias=False,
                      W_init=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')),\
                argscope([Conv2D, BatchNorm], data_format=self.data_format):
            i = tf.transpose(images, [0, 3, 1, 2])
            i = i if not self.input_norm else i / 255.0

            ####
            d = EfficientNetB0(i, self.freeze_en)

            np_feat = decoder('np', d)
            np_feat = tf.identity(np_feat, name='np_feat')
            npx = BNReLU('preact_out_np', np_feat)


            ordi_feat = decoder('ordi', d)
            ordi = BNReLU('preact_out_ordi', ordi_feat)

            if self.type_classification:
                tp_feat = decoder('tp', d)
                tp = BNReLU('preact_out_tp', tp_feat)

                # Nuclei Type Pixels (NT)
                logi_class = Conv2D('conv_out_tp', tp, self.nr_types, 1, use_bias=True, activation=tf.identity)
                logi_class = tf.transpose(logi_class, [0, 2, 3, 1])
                soft_class = tf.nn.softmax(logi_class, axis=-1)

            ### Nuclei Pixels (NF)
            logi_np = Conv2D('conv_out_np', npx, 2, 1, use_bias=True, activation=tf.identity)
            logi_np = tf.transpose(logi_np, [0, 2, 3, 1])
            soft_np = tf.nn.softmax(logi_np, axis=-1)
            prob_np = tf.identity(soft_np[...,1], name='predmap-prob-np')
            prob_np = tf.expand_dims(prob_np, axis=-1)


            ### Ordinal (NO)
            logi_ord = Conv2D('conv_out_ord', ordi, 16, 1, use_bias=True, activation=tf.identity)
            logi_ord_t = tf.transpose(logi_ord, [0, 2, 3, 1])
            N, C ,H, W = logi_ord.get_shape().as_list()
            ord_num = int(C/2)
            logi_ord = tf.reshape(logi_ord_t, shape=[-1, H, W, ord_num, 2])
            prob_ord = tf.nn.softmax(logi_ord, axis=-1)
            prob_ord = tf.identity(prob_ord, name='prob_ord')
            nn_out_labels = tf.reduce_sum(tf.argmax(prob_ord, axis=-1, output_type=tf.int32), axis=3, keepdims=True) 
            pred_ord = tf.identity(nn_out_labels, name='predmap-ord')
            
            ### encoded so that inference can extract all output at once
            predmap_coded = tf.concat([soft_class, prob_np], axis=-1, name='predmap-coded')
            

        
        

        def loss_ord(prob_ord, true_ord, name=None):
            (ord_n_pk, ord_pk) = tf.unstack(prob_ord, axis=-1)
            epsilon = tf.convert_to_tensor(10e-8, ord_n_pk.dtype.base_dtype)
            ord_n_pk, ord_pk = tf.clip_by_value(ord_n_pk, epsilon, 1 - epsilon), tf.clip_by_value(ord_pk, epsilon, 1 - epsilon)
            ord_log_n_pk = tf.log(ord_n_pk)
            ord_log_pk = tf.log(ord_pk)
            (N, H, W, C) = ord_log_pk.get_shape().as_list()
            foreground_mask = tf.reshape(tf.sequence_mask(true_ord, C), shape=[-1, H, W, C])
            sum_of_p = tf.reduce_sum(tf.where(foreground_mask, ord_log_pk, ord_log_n_pk), axis=3)
            loss = -tf.reduce_mean(sum_of_p)           
            loss = tf.identity(loss, name=name)
            return loss

        

        ####
        if get_current_tower_context().is_training:
            #---- LOSS ----#
            loss = 0
            
            for term, weight in self.loss_term.items():
                if term == 'bce':
                    term_loss = categorical_crossentropy_modified(soft_np, one_np)
                    term_loss = tf.reduce_mean(term_loss, name='loss-bce')
                elif term == 'dice':
                    term_loss = dice_loss(soft_np[...,0], one_np[...,0]) \
                                + dice_loss(soft_np[...,1], one_np[...,1])
                    term_loss = tf.identity(term_loss, name='loss-dice')
                elif term == 'ord':
                    term_loss = loss_ord(prob_ord, true_ord, name='loss-ord')
                else:
                    assert False, 'Not support loss term: %s' % term
                add_moving_summary(term_loss)
                loss += term_loss * weight
                
            if self.type_classification:
                term_loss = focal_loss_modified(soft_class, one_type)
                term_loss = tf.reduce_mean(term_loss, name='loss-classification')
                add_moving_summary(term_loss)
                loss = loss + term_loss

                


            self.cost = tf.identity(loss, name='overall-loss')            
            add_moving_summary(self.cost)
            ####

            add_param_summary(('.*/W', ['histogram']))   # monitor W
            
        return


class Sonnet_phase2(Model):

    def build_graph(self, inputs, truemap_coded):
        images = inputs
        orig_imgs = images

        if hasattr(self, 'type_classification') and self.type_classification:
            true_type = truemap_coded[..., 1]
            true_type = tf.cast(true_type, tf.int32)
            true_type = tf.identity(true_type, name='truemap-type')
            one_type = tf.one_hot(true_type, self.nr_types, axis=-1)
            true_type = tf.expand_dims(true_type, axis=-1)

            true_np = tf.cast(true_type > 0, tf.int32)
            true_np = tf.identity(true_np, name='truemap-np')
            one_np = tf.one_hot(tf.squeeze(true_np, axis=-1), 2, axis=-1)


        else:
            true_np = truemap_coded[..., 0]
            true_np = tf.cast(true_np, tf.int32)
            one_np = tf.one_hot(true_np, 2, axis=-1)
            true_np = tf.expand_dims(true_np, axis=-1)
            true_np = tf.identity(true_np, name='truemap-np')
        

        true_ord = truemap_coded[...,-1]
        true_ord = tf.expand_dims(true_ord, axis=-1)
        true_ord = tf.identity(true_ord, name='true-ord') 

        ####
        with argscope(Conv2D, activation=tf.identity, use_bias=False,
                      W_init=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')),\
                argscope([Conv2D, BatchNorm], data_format=self.data_format):
            i = tf.transpose(images, [0, 3, 1, 2])
            i = i if not self.input_norm else i / 255.0

            ####
            d = EfficientNetB0(i, self.freeze_en)

            np_feat = decoder('np', d)
            np_feat = tf.identity(np_feat, name='np_feat')
            npx = BNReLU('preact_out_np', np_feat)


            ordi_feat = decoder('ordi', d)
            ordi = BNReLU('preact_out_ordi', ordi_feat)

            if self.type_classification:
                tp_feat = decoder('tp', d)
                tp = BNReLU('preact_out_tp', tp_feat)

                # Nuclei Type Pixels (NT)
                logi_class = Conv2D('conv_out_tp', tp, self.nr_types, 1, use_bias=True, activation=tf.identity)
                logi_class = tf.transpose(logi_class, [0, 2, 3, 1])
                soft_class = tf.nn.softmax(logi_class, axis=-1)

            ### Nuclei Pixels (NF)
            logi_np = Conv2D('conv_out_np', npx, 2, 1, use_bias=True, activation=tf.identity)
            logi_np = tf.transpose(logi_np, [0, 2, 3, 1])
            soft_np = tf.nn.softmax(logi_np, axis=-1)
            prob_np = tf.identity(soft_np[...,1], name='predmap-prob-np')
            prob_np = tf.expand_dims(prob_np, axis=-1)


            ### Ordinal (NO)
            logi_ord = Conv2D('conv_out_ord', ordi, 16, 1, use_bias=True, activation=tf.identity)
            logi_ord_t = tf.transpose(logi_ord, [0, 2, 3, 1])
            N, C ,H, W = logi_ord.get_shape().as_list()
            ord_num = int(C/2)
            logi_ord = tf.reshape(logi_ord_t, shape=[-1, H, W, ord_num, 2])
            prob_ord = tf.nn.softmax(logi_ord, axis=-1)
            prob_ord = tf.identity(prob_ord, name='prob_ord')
            nn_out_labels = tf.reduce_sum(tf.argmax(prob_ord, axis=-1, output_type=tf.int32), axis=3, keepdims=True) # prediction output (sum to take the exact class, e.g.class 4)
            pred_ord = tf.identity(nn_out_labels, name='predmap-ord') # [N, 76, 76, 1]
            
            ### encoded so that inference can extract all output at once
            predmap_coded = tf.concat([soft_class, prob_np], axis=-1, name='predmap-coded')
            

        
        

        def loss_ord(prob_ord, true_ord, name=None):
            weight_map = tf.squeeze(true_ord, axis=-1)
            weight_map_1 = tf.where(
                tf.equal(weight_map, 1),
                2 * tf.cast(tf.equal(weight_map, 1), tf.float32),
                tf.cast(tf.equal(weight_map, 2), tf.float32)
            )
            (ord_n_pk, ord_pk) = tf.unstack(prob_ord, axis=-1)
            epsilon = tf.convert_to_tensor(10e-8, ord_n_pk.dtype.base_dtype)
            ord_n_pk, ord_pk = tf.clip_by_value(ord_n_pk, epsilon, 1 - epsilon), tf.clip_by_value(ord_pk, epsilon, 1 - epsilon)
            ord_log_n_pk = tf.log(ord_n_pk)
            ord_log_pk = tf.log(ord_pk)
            (N, H, W, C) = ord_log_pk.get_shape().as_list()
            foreground_mask = tf.reshape(tf.sequence_mask(true_ord, C), shape=[-1, H, W, C])
            sum_of_p = tf.reduce_sum(tf.where(foreground_mask, ord_log_pk, ord_log_n_pk), axis=3)
            sum_of_p += sum_of_p * weight_map_1
            loss = -tf.reduce_mean(sum_of_p)           
            loss = tf.identity(loss, name=name)
            return loss

        

        ####
        if get_current_tower_context().is_training:
            weight_map = tf.cast(tf.squeeze(pred_ord, axis=-1), tf.float32) # [N, 76, 76]
            weight_map_1 = tf.where(
                tf.equal(weight_map, 1),
                2 * tf.cast(tf.equal(weight_map, 1), tf.float32),
                tf.cast(tf.equal(weight_map, 2), tf.float32)
            )
            #---- LOSS ----#
            loss = 0
            
            for term, weight in self.loss_term.items():
                if term == 'bce':
                    term_loss = categorical_crossentropy_modified(soft_np, one_np)
                    term_loss += weight_map_1 * term_loss
                    term_loss = tf.reduce_mean(term_loss, name='loss-bce')
                elif term == 'dice':
                    term_loss = dice_loss(soft_np[...,0], one_np[...,0]) \
                                + dice_loss(soft_np[...,1], one_np[...,1])
                    term_loss = tf.identity(term_loss, name='loss-dice')
                elif term == 'ord':
                    term_loss = loss_ord(prob_ord, true_ord, name='loss-ord')
                else:
                    assert False, 'Not support loss term: %s' % term
                add_moving_summary(term_loss)
                loss += term_loss * weight
                
            if self.type_classification:
                term_loss = focal_loss_modified(soft_class, one_type)
                term_loss += weight_map_1 * term_loss
                term_loss = tf.reduce_mean(term_loss, name='loss-classification')
                add_moving_summary(term_loss)
                loss = loss + term_loss

            self.cost = tf.identity(loss, name='overall-loss')            
            add_moving_summary(self.cost)
            ####
        return