from multiprocessing import pool
from operator import index
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio 
import cv2
from matplotlib import cm
from torch_cluster import neighbor_sampler
import xlsxwriter
import glob
import os
import math
import pyheal
import skimage.filters
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import (distance_transform_cdt,
                                      distance_transform_edt)
import skimage.filters
from skimage.morphology import disk
import time
from matplotlib.colors import ListedColormap
import random
# from imgaug import augmenters as iaa
# import imgaug as ia
# from imgaug.augmentables.segmaps import SegmentationMapsOnImage
# ia.seed(4)
'''
Tumor: 60
EBV: 173
Benign: 157
'''

####
def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out 
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def rotate(image, angle, center=None, scale=1.0):
    h, w = image.shape[:2]
    if center is None:
        center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def size_calculate(major_id, inst_map):
    size_1 = np.sum((inst_map == (major_id + 1))>0)
    return size_1

def pick_minor_index(pool_minor, size_1, major_id=0):
    if major_id == 234:
        ann_2 = sio.loadmat('/media/tandoan/data2/Gastric_Cancer/Train/Labels/DB-0003_tumor_1.mat')
        basename = 'DB-0003_tumor_1'
        minor_class_id = 357
        return basename, ann_2, minor_class_id, pool_minor
    elif major_id == 29:
        ann_2 = sio.loadmat('/media/tandoan/data2/Gastric_Cancer/Train/Labels/DB-0017_3.mat')
        basename = 'DB-0017_3'
        minor_class_id = 281
        return basename, ann_2, minor_class_id, pool_minor

    for basename, minor_class_list in pool_minor.items():
        ann_2 = sio.loadmat('/media/tandoan/data2/Gastric_Cancer/Train/Labels/' + basename + '.mat')
        inst_map_2 = ann_2['inst_map']
        for minor_class_id in minor_class_list:
            mask_2 = (inst_map_2 == (minor_class_id+1)).astype(np.uint8)
            size_2 = np.sum(mask_2>0)
            # if size_1 >= 2.3*size_2:
            if size_1 >= 2.3 * size_2:
                pool_minor[basename].remove(minor_class_id)
                return basename, ann_2, minor_class_id, pool_minor

# file_1_list = glob.glob('/media/tandoan/data2/Gastric_Cancer/Train/Images/*.tif')
# file_1_list.remove('/media/tandoan/data2/Gastric_Cancer/Train/Images/DB-0003_tumor_1.tif')
# for file_1 in file_1_list:
#     print(file_1)
file_1 = '/media/tandoan/data2/Gastric_Cancer/Train/Images/DB-0003_tumor_1.tif'
eps = 5
img_list = glob.glob('/media/tandoan/data2/Gastric_Cancer/Train/Images/*.tif')
random.shuffle(img_list)
img_list.remove(file_1)

file_name = os.path.basename(file_1)
basename = file_name.split('.')[0]
img = cv2.imread('/media/tandoan/data2/Gastric_Cancer/Train/Images/' + file_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

ann = sio.loadmat('/media/tandoan/data2/Gastric_Cancer/Train/Labels/' + basename + '.mat')
inst_map = ann['inst_map']
inst_map_out = inst_map.copy()
type_map = ann['type_map']
class_arr = np.squeeze(ann['inst_type'])
class_arr[(class_arr == 1) | (class_arr == 2) | (class_arr == 9) | (class_arr == 10)] = 1
class_arr[(class_arr == 4) | (class_arr == 5) | (class_arr == 6) | (class_arr == 7)] = 2
class_arr[(class_arr == 8) | (class_arr == 3)] = 3
# class_arr[(class_arr == 3) | (class_arr == 4)] = 3
# class_arr[(class_arr == 5) | (class_arr == 6) | (class_arr == 7)] = 4
class_arr_copy = class_arr.copy()
# bbox_ann = ann['bbox'] # [y1, y2, x1, x2]
cent_ann = ann['inst_centroid'] # x, y
for i, cent in enumerate(cent_ann): # old_value = 30
    if ((cent[1] < 30) or 
            (cent[1] > (inst_map.shape[0]-30)) or 
            (cent[0] < 30) or 
            (cent[0] > (inst_map.shape[1]-30))):
        class_arr_copy[i] = 0
nuc_color = img * (inst_map[...,np.newaxis] > 0)
avg_color_1 = [
                np.sum(nuc_color[...,0]) / np.sum(nuc_color[...,0]>0), 
                np.sum(nuc_color[...,1]) / np.sum(nuc_color[...,1]>0), 
                np.sum(nuc_color[...,2]) / np.sum(nuc_color[...,2]>0)
            ]

major_class_idx = list(np.where(class_arr_copy == 2)[0]) + list(np.where(class_arr_copy == 3)[0]) 
                    # list(np.where(class_arr_copy == 4)[0])
picked_major_class = list(np.random.choice(major_class_idx, int(0.4 * len(major_class_idx)), replace=False))
picked_major_class = sorted(picked_major_class, key=lambda x: size_calculate(x, inst_map))
try:
    picked_major_class.remove(234)
except ValueError:
    pass  
try:
    picked_major_class.remove(29)
except ValueError:
    pass 
picked_major_class.insert(0, 234)
picked_major_class.insert(0, 29)

final = img.copy()
inpainted = img.copy()

pool_minor = {}
class_arr_2 = class_arr_copy.copy()
cent_ann_2 = cent_ann.copy()
minor_class_idx = list(np.where(class_arr_2 == 1)[0])

pool_minor[basename] = minor_class_idx
for file in img_list:
    file_name = os.path.basename(file)
    basename_1 = file_name.split('.')[0]
    ann_2 = sio.loadmat('/media/tandoan/data2/Gastric_Cancer/Train/Labels/' + basename_1 + '.mat')
    inst_map_2 = ann['inst_map']
    class_arr_2 = np.squeeze(ann_2['inst_type'])
    class_arr_2[(class_arr_2 == 1) | (class_arr_2 == 2) | (class_arr_2 == 9) | (class_arr_2 == 10)] = 1
    class_arr_2[(class_arr_2 == 4) | (class_arr_2 == 5) | (class_arr_2 == 6) | (class_arr_2 == 7)] = 2
    class_arr_2[(class_arr_2 == 8) | (class_arr_2 == 3)] = 3
    # class_arr[(class_arr == 3) | (class_arr == 4)] = 3
    # class_arr[(class_arr == 5) | (class_arr == 6) | (class_arr == 7)] = 4
    cent_ann_2 = ann_2['inst_centroid']
    for i, cent in enumerate(cent_ann_2):
        if ((cent[1] < 30) or 
                (cent[1] > (inst_map_2.shape[0]-30)) or 
                (cent[0] < 30) or 
                (cent[0] > (inst_map_2.shape[1]-30))):
            class_arr_2[i] = 0
    minor_class_idx = list(np.where(class_arr_2 == 1)[0])
    pool_minor[basename_1] = minor_class_idx

# mask_inpaint = np.zeros_like(inst_map)
# for major_class_idx in picked_major_class:
#     mask_inpaint += (inst_map == (major_class_idx+1)).astype(np.uint8)
# mask_inpaint = binary_dilation(mask_inpaint, iterations=2).astype(np.uint8)
# pyheal.inpaint(final, mask_inpaint, eps)
# pyheal.inpaint(inpainted, mask_inpaint, eps)

for major_class_idx in picked_major_class:
    mask_0 = (inst_map == (major_class_idx+1)).astype(np.uint8)
    
    mask = binary_dilation(mask_0, iterations=2).astype(np.uint8)
    cent1 = cent_ann[major_class_idx]
    bbox1 = bounding_box(mask)
    h1, w1 = bbox1[1] - bbox1[0], bbox1[3] - bbox1[2]
    size_1 = np.sum(mask>0)

    # try:
    basename_2, ann_2, index_2, pool_minor = pick_minor_index(pool_minor, size_1, major_id=major_class_idx)
    # except TypeError:
        # continue
    img_2_ori = cv2.imread('/media/tandoan/data2/Gastric_Cancer/Train/Images/' + basename_2 + '.tif')
    img_2_ori = cv2.cvtColor(img_2_ori, cv2.COLOR_BGR2RGB)
    img_2 = img_2_ori.copy()
    inst_map_2 = ann_2['inst_map']
    mask_2 = (inst_map_2 == (index_2+1)).astype(np.uint8)
    cent_ann_2 = ann_2['inst_centroid']
    cent_2 = cent_ann_2[index_2]
    bbox2 = bounding_box(mask_2)
    h2, w2 = bbox2[1] - bbox2[0], bbox2[3] - bbox2[2]

    img_2[...,0][mask_2 > 0] = (img_2_ori[...,0][mask_2 > 0] + avg_color_1[0]) / 2
    img_2[...,1][mask_2 > 0] = (img_2_ori[...,1][mask_2 > 0] + avg_color_1[1]) / 2
    img_2[...,2][mask_2 > 0] = (img_2_ori[...,2][mask_2 > 0] + avg_color_1[2]) / 2

    class_arr[major_class_idx] = 1
    pyheal.inpaint(final, mask, eps)
    pyheal.inpaint(inpainted, mask, eps)
    inst_map_out[inst_map == (major_class_idx+1)] = 0

    img_copy = img.copy()
    img_copy[bbox1[0]:bbox1[1], bbox1[2]:bbox1[3], :] = img_2[
                                                                int(np.round(cent_2[1])-h1/2):int(np.round(cent_2[1])+h1/2),
                                                                int(np.round(cent_2[0])-w1/2):int(np.round(cent_2[0])+w1/2), 
                                                                :
                                                            ]
    mask_translated = np.zeros_like(mask)
    mask_translated[int(np.round(cent1[1])-h2/2):int(np.round(cent1[1])+h2/2), 
                    int(np.round(cent1[0])-w2/2):int(np.round(cent1[0])+w2/2)] = mask_2[bbox2[0]:bbox2[1], bbox2[2]:bbox2[3]]
    inst_map_out[mask_translated > 0] = major_class_idx + 1
    mask = ((mask + mask_translated)>0).astype(np.uint8)
    mask_substract = mask - mask_translated
    cdt_map = distance_transform_cdt(1 - mask_translated).astype('float32')
    cdt_map[mask==0] = 0
    cdt_map[mask_substract>0] -= 1
    cdt_map[mask_substract>0] /= np.amax(cdt_map[mask_substract>0])
    cdt_map[mask_substract>0] = 1 - cdt_map[mask_substract>0]
    cdt_map[mask_translated > 0] = 1
    plt.figure(dpi=400, figsize=(8.0, 8.0))
    plt.gca().set_axis_off()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.tight_layout()
    jet = cm.get_cmap('viridis', 255)
    jet = jet(np.linspace(0, 1, 255))
    jet[0, :] = np.array([0/256, 0/256, 0/256, 1])
    # jet[1, :] = np.array([255/256, 0/256, 0/256, 1])
    jet = ListedColormap(jet)
    plt.imshow(cdt_map[659:659+100, 1:1+100], cmap=jet)
    plt.savefig("/media/tandoan/data2/Gastric_Cancer/Train/Synthesized_Overlay/_zoom_cdt_1.tif", bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    # final = final*(1-mask[...,np.newaxis]) + img_copy*mask_translated[...,np.newaxis] + (img_copy*(cdt_map*mask_substract)[...,np.newaxis]).astype(np.uint8) + (final*((1-cdt_map)*mask_substract)[...,np.newaxis]).astype(np.uint8)
    final = (img_copy * cdt_map[...,np.newaxis]).astype(np.uint8) + (final * (1 - cdt_map)[...,np.newaxis]).astype(np.uint8)
    # smooth_synth = skimage.filters.gaussian(final, sigma=(2.0, 2.0), truncate=3.5, multichannel=True, preserve_range=True).astype(np.uint8)
    final_smooth = np.stack([skimage.filters.median(final[...,0], disk(1)), skimage.filters.median(final[...,1], disk(1)), skimage.filters.median(final[...,2], disk(5))], axis=2)
    final = (final * (1 - mask_substract[...,np.newaxis])).astype(np.uint8) + (final_smooth.astype(np.float32) * mask_substract[...,np.newaxis]).astype(np.uint8)
    
final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
type_map = np.zeros_like(type_map)
inst_list = list(np.unique(inst_map_out))
inst_type = []
inst_list.remove(0)
for inst_id in inst_list:
    type_map[inst_map_out == int(inst_id)] = class_arr[int(inst_id) - 1]
    inst_type.append(class_arr[int(inst_id-1)])
# cv2.imwrite('/media/tandoan/data2/Gastric_Cancer/Train/Synthesized_Images/' + basename + '_synthesized.tif', final)
# cv2.imwrite('/media/tandoan/data2/Gastric_Cancer/Train/Inpainted/' + basename + '_inpainted.tif', inpainted)
# sio.savemat('/media/tandoan/data2/Gastric_Cancer/Train/Synthesized_Labels/' + basename + '_synthesized.mat', 
#                 {'inst_map'  :     inst_map_out,
#                     'type_map'  :     type_map,
#                     'inst_type' :     np.array(class_arr[:, None]), 
#                     'inst_centroid' : cent_ann,
#                 })


# major_class_len = len(picked_major_class) - int(0.6*len(picked_major_class))
# j = 0
# count = 0
# while j < major_class_len:
#     count += 1
#     major_class_idx = picked_major_class[int(0.6*len(picked_major_class))+j]
#     mask_0 = (inst_map == (major_class_idx + 1)).astype(np.uint8)
    
    
#     mask = binary_dilation(mask_0, iterations=2).astype(np.uint8)
#     cent1 = cent_ann[major_class_idx]
#     bbox1 = bounding_box(mask)
#     h1, w1 = bbox1[1] - bbox1[0], bbox1[3] - bbox1[2]

#     img_2_path = np.random.choice(img_list, 1)[0]
#     while img_2_path == file:
#         img_2_path = np.random.choice(img_list, 1)[0]
#     file_name_2 = os.path.basename(img_2_path) 
#     basename_2 = file_name_2.split('.')[0]
#     img_2_ori = cv2.imread(img_2_path)
#     img_2_ori = cv2.cvtColor(img_2_ori, cv2.COLOR_BGR2RGB)
#     img_2 = img_2_ori.copy()
#     img_2[...,0] = (img_2_ori[...,0] + avg_color_1[0]) / 2
#     img_2[...,1] = (img_2_ori[...,1] + avg_color_1[1]) / 2
#     img_2[...,2] = (img_2_ori[...,2] + avg_color_1[2]) / 2
#     ann_2 = sio.loadmat('/media/tandoan/data2/Gastric_Cancer/Train/Labels/' + basename_2 + '.mat')
#     inst_map_2 = ann_2['inst_map']
#     class_arr_2 = np.squeeze(ann_2['inst_type'])
#     class_arr_2[(class_arr_2 == 1) | (class_arr_2 == 2) | (class_arr_2 == 9) | (class_arr_2 == 10)] = 1
#     class_arr_2[(class_arr_2 == 4) | (class_arr_2 == 5) | (class_arr_2 == 6) | (class_arr_2 == 7)] = 2
#     class_arr_2[(class_arr_2 == 8) | (class_arr_2 == 3)] = 3
#     cent_ann_2 = ann_2['inst_centroid']
#     for i, cent in enumerate(cent_ann_2):
#         if ((cent[1] < 30) or 
#                 (cent[1] > (inst_map_2.shape[0]-30)) or 
#                 (cent[0] < 30) or 
#                 (cent[0] > (inst_map_2.shape[1]-30))):
#             class_arr_2[i] = 0
#     minor_class_idx = list(np.where(class_arr_2 == 1)[0])
#     index_2 = np.random.choice(minor_class_idx, 1)[0]
#     mask_2 = (inst_map_2 == (index_2 + 1)).astype(np.uint8)
#     cent_2 = cent_ann_2[index_2]
#     bbox2 = bounding_box(mask_2)
#     h2, w2 = bbox2[1] - bbox2[0], bbox2[3] - bbox2[2]

#     size_1 = np.sum(mask>0)
#     size_2 = np.sum(mask_2>0)
#     if count > 7:
#         j += 1
#         continue
#     if size_1 < 2.3*size_2:
#         continue
#     class_arr[major_class_idx] = 1
#     inst_map_out[inst_map == (major_class_idx+1)] = 0
#     pyheal.inpaint(final, mask, eps)
#     j += 1
#     count = 0
#     img_copy = img.copy()
#     img_copy[bbox1[0]:bbox1[1], bbox1[2]:bbox1[3], :] = img_2[
#                                                                 int(np.round(cent_2[1])-h1/2):int(np.round(cent_2[1])+h1/2),
#                                                                 int(np.round(cent_2[0])-w1/2):int(np.round(cent_2[0])+w1/2), 
#                                                                 :
#                                                                 ]
#     mask_translated = np.full_like(mask, 0)
#     mask_translated[int(np.round(cent1[1])-h2/2):int(np.round(cent1[1])+h2/2), 
#                     int(np.round(cent1[0])-w2/2):int(np.round(cent1[0])+w2/2)] = mask_2[bbox2[0]:bbox2[1], bbox2[2]:bbox2[3]]
#     inst_map_out[mask_translated > 0] = major_class_idx + 1
#     mask_substract = mask - mask_translated
#     cdt_map = distance_transform_cdt(mask).astype('float32')
#     cdt_map = cdt_map / np.amax(cdt_map)
#     final = final*(1-mask[...,np.newaxis]) + img_copy*mask_translated[...,np.newaxis] + (img_copy*(cdt_map*mask_substract)[...,np.newaxis]).astype(np.uint8) + (final*((1-cdt_map)*mask_substract)[...,np.newaxis]).astype(np.uint8)

# final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
# cv2.imwrite('/media/tandoan/data2/Gastric_Cancer/Train/Synthesized_Images/' + basename + '_synthesized.tif', final)
# type_map = np.zeros_like(type_map)
# inst_list = list(np.unique(inst_map_out))
# inst_type = []
# inst_list.remove(0)
# for inst_id in inst_list:
#     type_map[inst_map_out == inst_id] = class_arr[inst_id - 1]
#     inst_type.append(class_arr[inst_id-1])

# sio.savemat('/media/tandoan/data2/Gastric_Cancer/Train/Synthesized_Labels/' + basename + '_synthesized.mat', 
#                 {'inst_map'  :     inst_map_out,
#                     'type_map'  :     type_map,
#                     'inst_type' :     np.array(class_arr[:, None]), 
#                     'inst_centroid' : cent_ann,
#                 })



