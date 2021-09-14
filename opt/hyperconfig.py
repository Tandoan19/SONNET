import tensorflow as tf

sonnet = {
    'train_input_shape' : [270, 270],
    'train_mask_shape'  : [76, 76],
    'infer_input_shape' : [270, 270],
    'infer_mask_shape'  : [76, 76], 

    'training_phase'    : [
        {
            'nr_epochs': 50,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (1.0e-4, [('25', 1.0e-5)]), 
            },
            'pretrained_path'  : './ImageNet_pretrained_EfficientB0.npz',
            'train_batch_size' : 8,
            'infer_batch_size' : 16,

            'model_flags' : {
                'freeze_en' : True
            }
        },

        {
            'nr_epochs': 25,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (1.0e-4, [('25', 1.0e-5)]), 
            },
            # path to load, -1 to auto load checkpoint from previous phase, 
            # None to start from scratch
            'pretrained_path'  : -1,
            'train_batch_size' : 4,
            'infer_batch_size' : 8,

            'model_flags' : {
                'freeze_en' : False
            }
        },
        
        {
            'nr_epochs': 25,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (1.0e-5, [('25', 1.0e-5)]), 
            },
            # path to load, -1 to auto load checkpoint from previous phase, 
            # None to start from scratch
            'pretrained_path'  : -1,
            'train_batch_size' : 4, 
            'infer_batch_size' : 8,

            'model_flags' : {
                'freeze_en' : False
            }
        }
    ],

    'loss_term' : {'bce' : 1, 'dice' : 1}, 

    'train_optimizer'           : tf.train.AdamOptimizer,

    'inf_auto_metric'   : 'valid_dice',
    'inf_auto_comparator' : '>',
    'inf_batch_size' : 4,
}