#load the mapping network, pass 100,000 latents and record the mean/standard deviation and save it as file
import copy
import os
from time import perf_counter

import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import sys

import dnnlib
import legacy
# from baseline.stylegan3projector import dnnlib
# from baseline.stylegan3projector import legacy
# #sys.path.append('./baseline/stylegan3-projector')
# #import dnnlib
# # import baseline/dnnlib
# # import baseline/legacy


source = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/'
network_pkl = 'stylegan3-r-ffhqu-1024x1024.pkl'
with dnnlib.util.open_url(source+network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False) # type: ignore

map_net = G.mapping
latent = np.random.RandomState(123).randn(10000, 512)
w_samples =  map_net(torch.from_numpy(latent), None)
cor_latent = torch.nn.LeakyReLU(5)(w_samples)[:,0,:]
latent_mean = cor_latent.mean(0)
latent_std = cor_latent.std(0)
gaussian_fit = {"mean": latent_mean, "std": latent_std}
torch.save(gaussian_fit,"gaussian_fit.pt")