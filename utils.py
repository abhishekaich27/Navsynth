from os.path import isfile, isdir, join
import torch
import numpy as np
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as fnn


def sample_z_motion(num_samples, dim_z_motion):
    eps = torch.randn(dim_z_motion, num_samples)
    
    return torch.autograd.Variable(torch.FloatTensor(eps).cuda(), requires_grad=True)

def sample_z_content(dim_z_content):
    z_c = np.random.normal(0, 1, (dim_z_content, 1)).astype(np.float32)  

    return torch.autograd.Variable(torch.FloatTensor(z_c).cuda(), requires_grad=True) 

def repeat_z_c(z_c, video_len):
    c = z_c.repeat(1, video_len)

    return torch.autograd.Variable(c.cuda(), requires_grad=True) 

def join_z(z_c, z_motion, video_len): 
    z_content = repeat_z_c(z_c, video_len)  
    z = torch.cat([z_content, z_motion], dim=0)
    norm = z.norm(p=2, dim=1, keepdim=True)
    z = z.div(z.max().expand_as(z))

    return z

def join_z_epoch(z_c, z_motion, video_len):   
    z = torch.cat([z_c, z_motion], dim=0)
    norm = z.norm(p=2, dim=1, keepdim=True)

    z = z.div(z.max().expand_as(z))

    return z

def create_corr(eps, gru, video_len, dim_z_m):
    gru.initHidden(video_len)
    z_m = gru(eps.reshape(-1, dim_z_m), video_len)

    return z_m.view(-1, video_len)

##### Loss functions 
def l2_loss(x_hat, x):
    return (x_hat - x).pow(2).mean()

def l1_loss(x_hat, x):
    return abs(x_hat - x).mean()

def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)

    kernel = np.tile(kernel, (n_channels, 1, 1))
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)

def conv_gauss(img, kernel):
    n_channels, _, kw, kh = kernel.shape
    img = fnn.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')

    return fnn.conv2d(img, kernel, groups=n_channels)

def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = fnn.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr

class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None
        
    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(
                size=self.k_size, sigma=self.sigma, 
                n_channels=input.shape[1], cuda=input.is_cuda
            )
        pyr_input  = laplacian_pyramid( input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(fnn.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

def compute_PSNR(x, x_hat):
    MSE = np.mean((x_hat-x)**2)
    PSNR = 20*np.log10((np.max(x)-np.min(x))/np.sqrt(MSE))
    
    return PSNR
