from pickletools import uint8
import torch
import numpy as np
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import lpips
import config
from preprocess import prep_image
import dnnlib
import legacy
device = 'cpu'


class projector():
    def __init__(self):
        self.p_im = prep_image()
        self.p_im.corrupt_image()

    def load_network(self):
        source = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/'
        network_pkl = 'stylegan3-r-ffhqu-1024x1024.pkl'
        with dnnlib.util.open_url(source+network_pkl) as fp:
                G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False) # type: ignore
        return G

    def project(self):
        target_img = self.p_im.corrupted_img.astype('uint8')
        corr_im = Image.fromarray(target_img)
        corr_im.save('corrupted_img.jpg')
        target_img = torch.from_numpy(target_img).to(device)

        ###prep target image/normalized
        

        ###load generator network
        generator = self.load_network().to(device)

        ###initialize latent variables
        latent = torch.randn((1,16,512),dtype=torch.float,requires_grad=True,device=device)

        ###initialize noise variables
        noise_vars = {name:buf for (name,buf) in generator.synthesis.named_buffers() }
        #print(noise_vars)
        # for i in noise_vars:
        #     print(i)
        print(generator.w_dim)
        out = generator.synthesis(latent)
        print(out.shape)



p = projector()
p.project()


