from glob import glob
from scipy import misc, linalg
import os, torch
from os.path import isfile, isdir, join
from os import mkdir
import matplotlib
matplotlib.use('tkagg')
import numpy as np

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import train, models, utils, get_plot
import gc


## Network parameters
ngf = 64 # Height
ndf = 64 # Width
nc  = 3  # Channel
ngpu = 2 # Num of GPU

## Number of training epochs for generator, RNN and latent codes
num_epoch = 5    # Stage 1
num_code = 300   # Stage 2

## Latent dimensions
z_dim = 256          # Total latent dimension
dim_z_motion = 200   # Total transient latent dimension
dim_z_content = z_dim - dim_z_motion # Total static latent dimension

def train_class(video_path, fc, eps, video_idx, test_epochs, nc, ngf, ndf, ngpu, gen, rnn, fc_path,z_dim, dim_z_motion):
    video_path = video_path + '/'
    sequence_dir = video_path
    sequence_files = sorted(glob(sequence_dir+'*.png'))
     
    sequence_size = len(sequence_files)

    x_test = []

    for i in range (0, sequence_size):
        if i > 15:
           continue
        img = misc.imread(sequence_files[i])
        img = img[100:500, 100:500, :]
        img = misc.imresize(img, [ngf, ndf])
        img = img/255.0

        temp = np.zeros((3, img.shape[0], img.shape[1]))    
        for chan in range (0, temp.shape[0]):
            temp[chan, :, :] = img[: , :, chan]     
        
        x_test.append(temp)
    x_test = np.array(x_test) 

    psnr = train.train(x_test, 16, video_path, fc, eps, test_epochs, video_idx, nc, ngf, ndf, ngpu, gen, rnn, fc_path, z_dim, dim_z_motion)
 
    return psnr


## Get all video paths and store as a list
data_folder_path = 'path_to_training_data' # Enter the path to downloaded dataset
folders = []

for dirpath, dirs, filenames in os.walk(data_folder_path):
    folders.append(dirpath) 

delete_ = []
for path in folders:
    if not path.endswith('renders'):
       delete_.append(path)

data_folders = list(set(folders)^set(delete_))

## Create latent code for each video
def makedirFc(fc_dir, video_path):
    fc_path = os.path.join(fc_dir, video_path.split('/')[-2])
    
    if not os.path.exists(fc_path):
        os.makedirs(fc_path)

        #Randomly gen z_c and store
        eps = utils.sample_z_motion(16, dim_z_motion) 

        #store 
        fc_path = os.path.join(fc_path, 'eps.pth')
        torch.save(eps, fc_path) 
    else:
        fc_path = os.path.join(fc_path,'eps.pth')

    return torch.load(fc_path), fc_path 

fc_dir = './fc_dir/'

## Initialize Generator, RNN, and latent codes 

generator = models.Generator(ngpu, z_dim, ngf, ndf, nc)
generator = generator.cuda()

gru = models.GRU(dim_z_motion, 500, gpu = True)
gru.initWeight()
gru = gru.cuda()

z_c = utils.sample_z_content(dim_z_content) 


## Start training

for ep in range(num_epoch):

    #Random shuffle data_folder
    np.random.shuffle(data_folders)
   
    train_tqdm = tqdm(range(100))
    psnr = 0.0
    
    for i in train_tqdm:
        gru.train()
        generator.train()

        video_path = data_folders[i]
        eps, fc_path = makedirFc(fc_dir, video_path) 
        
        psnr += train_class(video_path, eps, z_c, i, num_code, nc, ngf, ndf, ngpu, generator, gru, fc_path, z_dim, dim_z_motion)
        
        train_tqdm.set_description('Epoch {} Avg. PSNR {}'.format(ep,1.0*psnr/(i+1)))
        train_tqdm.refresh()
    train_tqdm.close()
   



