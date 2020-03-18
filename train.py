from glob import glob
from scipy import misc, linalg
import torch
import torch.nn as nn
from os.path import isfile, isdir, join
from torch.autograd import Variable
from tqdm import tqdm
from os import mkdir
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

import models
import triplet_loss
import utils
import get_plot 
import os

def checkDirExists(d_path):
    if not os.path.exists(d_path):
        os.makedirs(d_path)

def train(x_test, sequence_size, video_path, eps, z_c, test_epochs, video_idx, nc, ngf, ndf, ngpu, generator, gru, fc_path, z_dim, dim_z_motion):
     
     device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu") 
     
     ## Store a copy of x_test
     x_test_org = x_test.copy()
     
     ## Denorm the images
     x_test=2*x_test_org-1

     ##### Input options

     test_size = sequence_size
     test_batch_size = sequence_size
 
     lr = 0.001*test_batch_size/256.0         # Learning rate for Generator weights
     lr_gru = 0.001*test_batch_size/256.0       # Learning rate for RNN weights
     alpha = 200.0*test_batch_size/256.0      # Learning rate for latent codes

     lambda_image = 0.01
     
     dim_z_content = z_dim - dim_z_motion

     #### Optimization

     batch_no = 1 
     idx = np.arange(test_size)
     loss_test = []
     
     #z_c_batch = utils.repeat_z_c(z_c, sequence_size)

     optimizer_net = torch.optim.Adam([
             {'params': generator.parameters(), 'lr': lr},
             {'params': gru.parameters(), 'lr': lr_gru}
    
     ])
     optimizer_lat = torch.optim.SGD([
             {'params': eps, 'lr': alpha},
             {'params': z_c, 'lr': alpha}
    
     ])
     
     loss_func = utils.LapLoss(max_levels = 4).cuda()
     
     #### Start training 

     for batch_idx in range(0, batch_no):
         
         l_lim = 0 
         u_lim = l_lim + test_batch_size 
        
         x_batch = x_test[l_lim:u_lim, :, :, :]
         x_batch_tensor = torch.cuda.FloatTensor(x_batch).view(-1, nc, ngf, ndf)
     
         eps_batch = eps[:,l_lim:u_lim].cuda()
   
         batch_tqdm = range (0, test_epochs)
         loss_epoch=[]

         for epoch in batch_tqdm:
             gru.train()
             generator.train() 

             #print(epoch)
             ''' ********************************* Part 1 *********************** '''
             z_c_batch = utils.repeat_z_c(z_c, sequence_size)
         
             ## Prepare motion latent code from RNN 
             z_m = utils.create_corr(eps_batch, gru, u_lim-l_lim, dim_z_motion)
             
             ## Concatinate motion and content code
             z_batch = utils.join_z_epoch(z_c_batch, z_m, test_batch_size)
             
             z_batch_tensor = z_batch.view(-1, z_dim, 1, 1)
             
             ## Predict the video and compute loss
             x_hat = generator(z_batch_tensor) 
             loss_video = loss_func(x_hat, x_batch_tensor)

             ## Predict a random frame and compute loss  
             i = np.random.randint(u_lim-l_lim, size=1)
             loss_image = loss_func(x_hat[i, :,:,:], x_batch_tensor[i, :,:,:])
             
             ## Learning loss
             loss = loss_video + lambda_image*loss_image
             loss_epoch.append(loss.item())
        
             optimizer_net.zero_grad()
             optimizer_lat.zero_grad() 
             loss.backward(retain_graph = True) 
             optimizer_net.step()
             optimizer_lat.step()

             ''' ********************************* Part 2 *********************** '''
             ## Apply triplet loss to motion latent code from RNN
             loss_triplet = 0.01*triplet_loss.triplet_cond(z_m, 2, test_batch_size, dim_z_motion)
             
             optimizer_net.zero_grad()
             optimizer_lat.zero_grad() 
             loss_triplet.backward() 
             optimizer_net.step()
             optimizer_lat.step()
             

         loss_test.append(np.array(loss_epoch))
    ##  Test the model performance
     for i in range(1):
         with torch.no_grad():

                  ## Sample for video generation
                  eps_gen_tensor = torch.cuda.FloatTensor(eps_batch)
                  gru.eval()
                  generator.eval()

                  z_c_batch = utils.repeat_z_c(z_c, sequence_size)
                  z_m_gen = utils.create_corr(eps_gen_tensor, gru, sequence_size, dim_z_motion)
                  z = utils.join_z_epoch(z_c_batch, z_m_gen, sequence_size)
    
                  x_gen_test = generator(z.view(-1, z_dim, 1, 1))
                  x_gen = x_gen_test/2+0.5 
    
                  ## Save latent codes and models
                  image_path = './videos/'
                  mp4_path = './mp4_files/'      
                  
                  get_plot.save_images(x_gen.cpu(), 16, image_path + video_path.split('/')[-3])
                  get_plot.get_video(x_gen.detach().cpu().numpy(), 16, mp4_path + video_path.split('/')[-3] +'.mp4')

                  torch.save(eps, fc_path)
                  torch.save(z_c, path + './content.pt')
                  torch.save(generator.state_dict(), path + './generator.pth')
                  torch.save(gru.state_dict(), path + './gru.pth')

                  psnr = utils.compute_PSNR(x_test_org, x_gen.detach().cpu().numpy())

     return psnr
