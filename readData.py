from __future__ import print_function
import os
import math
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.ndimage as ndimage

target_height = 512
target_width = 256


class ListDatasetFloatMat_cw():

    def __init__(self, data_folder, list_file, transform=None):
        '''
        Args:
          mat_data: (str) image mat cells.
          list_file: (str/[str]) path to index file.
          transform: (function) image/box transforms.
        '''
        self.image_path_list = []
        self.transform = transform

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split()
            file_path = os.path.join(data_folder, splited[0])
            self.image_path_list.append(file_path)

    def _apply_transforms(self, image):
        for t in self.transform:
            image = t(image)
        return image

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.
        '''

        # Load image data
        file_path = self.image_path_list[idx]
        flnm = os.path.basename(file_path)
        # print(file_path)
        tmpmat = sio.loadmat(file_path)
        label = tmpmat['label']
        label = np.array(label).astype(float)

        # data
        cw = tmpmat['cw']
        cw = np.array(cw).astype(float)
        rect_cw = np.array(tmpmat['rect_cw'])
        cw_baselayer = ndimage.filters.gaussian_filter(cw, 64, order=0)
        cw = cw - cw_baselayer
        cw_std = np.std(cw)
        cw = cw / cw_std

        crop_cw = cw[rect_cw[0, 0]:rect_cw[0, 1], rect_cw[0, 2]:rect_cw[0, 3]]

        high, wid = crop_cw.shape
        ori_ratio = high / wid
        if ori_ratio > 2:
            h_scale = (target_height / high)
            w_scale = target_height / ori_ratio / wid
        else:
            w_scale = (target_width / wid)
            h_scale = target_width * ori_ratio / high

        # h_scale, w_scale = (target_height / high), (target_width / wid)
        crop_cw = ndimage.zoom(crop_cw, (h_scale, w_scale))

        # padding
        zoom_high, zoom_wid = crop_cw.shape
        pad_high, pad_wid = target_height - zoom_high, target_width - zoom_wid
        top = int(np.floor(pad_high / 2))
        bottom = pad_high - top
        left = int(np.floor(pad_wid / 2))
        right = pad_wid - left
        minVal = np.min(crop_cw)
        crop_cw = np.pad(crop_cw, ((top, bottom), (left, right)), 'constant',  constant_values=minVal)

        if self.transform:
            crop_cw = self._apply_transforms(crop_cw)

        img = crop_cw.copy()
        # print(img.shape)

        return img, flnm, label

    def __len__(self):
        return self.num_imgs

class ListDatasetFloatMat_zw():

    def __init__(self, data_folder, list_file, transform=None):
        '''
        Args:
          mat_data: (str) image mat cells.
          list_file: (str/[str]) path to index file.
          transform: (function) image/box transforms.
        '''
        self.image_path_list = []
        self.transform = transform

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split()
            file_path = os.path.join(data_folder, splited[0])
            self.image_path_list.append(file_path)

    def _apply_transforms(self, image):
        for t in self.transform:
            image = t(image)
        return image

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.
        '''

        # Load image data
        file_path = self.image_path_list[idx]
        flnm = os.path.basename(file_path)
        # print(file_path)
        tmpmat = sio.loadmat(file_path)
        label = np.array(label).astype(float)

        # data
        cw = tmpmat['zw']
        cw = np.array(cw).astype(float)
        rect_cw = np.array(tmpmat['rect_zw_pred'])
        cw_baselayer = ndimage.filters.gaussian_filter(cw, 64, order=0)
        cw = cw - cw_baselayer
        cw_std = np.std(cw)
        cw = cw / cw_std

        crop_cw = cw[rect_cw[0, 0]:rect_cw[0, 1], rect_cw[0, 2]:rect_cw[0, 3]]

        high, wid = crop_cw.shape
        ori_ratio = high / wid
        if ori_ratio > 2:
            h_scale = (target_height / high)
            w_scale = target_height / ori_ratio / wid
        else:
            w_scale = (target_width / wid)
            h_scale = target_width * ori_ratio / high

        # h_scale, w_scale = (target_height / high), (target_width / wid)
        crop_cw = ndimage.zoom(crop_cw, (h_scale, w_scale))

        # padding
        zoom_high, zoom_wid = crop_cw.shape
        pad_high, pad_wid = target_height - zoom_high, target_width - zoom_wid
        top = int(np.floor(pad_high / 2))
        bottom = pad_high - top
        left = int(np.floor(pad_wid / 2))
        right = pad_wid - left
        minVal = np.min(crop_cw)
        crop_cw = np.pad(crop_cw, ((top, bottom), (left, right)), 'constant',  constant_values=minVal)

        if self.transform:
            crop_cw = self._apply_transforms(crop_cw)

        img = crop_cw.copy()
        # print(img.shape)

        return img, flnm, label

    def __len__(self):
        return self.num_imgs




class ListData4Seg():

    def __init__(self, data_folder, list_file, use_cw=False, crop=False, scale=True, transform=None):
        '''
        Args:
          mat_data: (str) image mat cells.
          list_file: (str/[str]) path to index file.
          transform: (function) image/box transforms.
        '''
        self.image_path_list = []
        self.transform = transform
        self.use_cw = use_cw
        self.crop = crop
        self.scale = scale

        self.seen = 0
        self.shape = 512

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split()
            file_path = os.path.join(data_folder, splited[0])
            self.image_path_list.append(file_path)

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.
        '''

        # Load image data
        file_path = self.image_path_list[idx]
        filename = os.path.basename(file_path)
        tmpmat = sio.loadmat(file_path)

        if self.use_cw:
            cw = tmpmat['cw']
            cw_mask = tmpmat['cw_mask']
            cw = np.array(cw).astype(float)
            cw_baselayer = ndimage.filters.gaussian_filter(cw, 64, order=0)
            cw = cw - cw_baselayer
            cw_std = np.std(cw)
            cw = cw / cw_std

            if self.scale:
                high1, wid1 = cw.shape
                h_scale1, w_scale1 = (target_height / high1), (target_width / wid1)
                # h_scale1, w_scale1 = (target_height / high1), (target_width / wid1)
                cw = ndimage.zoom(cw, (h_scale1, w_scale1))
                cw_mask = ndimage.zoom(cw_mask, (h_scale1, w_scale1))
                cw_mask = cw_mask > 0

        else:
            # data
            zw = tmpmat['zw']
            zw = np.array(zw).astype(float)
            zw_mask = tmpmat['zw_mask']
            zw_baselayer = ndimage.filters.gaussian_filter(zw, 64, order=0)
            zw = zw - zw_baselayer
            zw_std = np.std(zw)
            zw = zw / zw_std

            if self.scale:
                high, wid = zw.shape
                h_scale, w_scale = (target_height / high), (target_width / wid)
                zw = ndimage.zoom(zw, (h_scale, w_scale))
                zw_mask = ndimage.zoom(zw_mask, (h_scale, w_scale))
                zw_mask = zw_mask > 0

        if self.use_cw:
            return cw, cw_mask, filename
        else:
            return zw, zw_mask, filename

    def __len__(self):
        return self.num_imgs


