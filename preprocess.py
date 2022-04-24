import numpy as np
import config
import torch
import cv2
import sys
from PIL import Image

class prep_image():
    def __init__(self):
        self.original_img = np.array(Image.open(config.optimization['input_file']).convert('RGB'))
        self.original_img = cv2.resize(self.original_img,(1024,1024))
        self.corrupted_img = None
        self.inpaint_mask = None
        self.resolution_factor = None
    def corrupt_image(self):
        corruption = config.optimization['corruption']
        if corruption == 'random_inpaint':
            pct = 1-config.optimization['random_inpainting_percentage']
            #mask = np.random.binomial(1, pct, self.original_img.shape[:-1])
            mask = np.random.binomial(1,pct, (256,256)).repeat(4,axis=0).repeat(4,axis=1)
            self.inpaint_mask = mask
            self.corrupted_img = np.einsum('ijk, ijk -> ijk', self.original_img, mask[..., np.newaxis])
        if corruption == 'inpaint':
            self.corrupted_img = self.original_img.copy()
            self.corrupted_img[412:612, 312:712,:] = 0
        if corruption == 'super-res':
            pct = 1/config.optimization['super_res']
            self.corrupted_img = cv2.resize(self.original_img, (0,0), fx=pct, fy=pct)
        if corruption == 'denoising':
            std = config.optimization['noise_std']
            mask = np.random.normal(loc=0.0, scale=std, size=self.original_img.shape)
            self.corrupted_img = np.clip(self.original_img + mask, 0, 255).astype('int')
        if corruption == 'colorization':
            self.corrupted_img = self.original_img.copy()
            self.corrupted_img[:,:] = self.corrupted_img.mean(axis=-1,keepdims=1)
