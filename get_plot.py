import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import utils as vu
from os.path import isfile, isdir, join
import cv2


## Function to see the input video
def show_images(normed, view_frames):
    normed = torch.FloatTensor(normed).narrow(0, 0, view_frames)
    im = vu.make_grid(normed, nrow=view_frames).numpy()
    im = im.transpose((1, 2, 0))
    plt.axis('off')
    plt.imshow(im)
    plt.show(block=True)

## Function to save the input video as a frame sequence jpg
def save_images(normed, view_frames, name):
    normed = torch.FloatTensor(normed).narrow(0, 0, view_frames)
    vu.save_image(normed, name+'.jpg', nrow=view_frames)

## Function to save the input video as mp4 file
def get_video(img_arr, video_len, name):

    frames = img_arr.transpose((0, 2, 3, 1))
    _, h, w, _ = frames.shape
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), 8, (w, h))

    for i in range(video_len):
        imtemp = ((frames[i, :, :, :]*255).copy()).astype('uint8')
        out.write(imtemp)
    out.release()
    
    return 
