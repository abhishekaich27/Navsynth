# Non-Adversarial Video Synthesis with Learned Priors

## Overview
This package is a PyTorch implementation of the paper "Non-Adversarial Video Synthesis with Learned Priors" accepted at [IEEE CVPR 2020](http://cvpr2020.thecvf.com/).</br> [[Project page]](https://abhishekaich27.github.io/navsynth.html) [[Paper]](https://arxiv.org/abs/2003.09565)

## Dependencies
Create a conda environment with the 'environment.yml'

## Data
Please download the [Chair-CAD](https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar) dataset.

## Running
1. Create folders with names "fc_dir", "videos" and "mp4_files".
2. Set the dataset path in the "class_gen.py" file.
3. Run the following:
```javascript 
python class_gen.py
```

## Citation
Please cite the following work if you use this package.
```javascript
@InProceedings{Aich_2020_CVPR,
author = {Aich, Abhishek and Gupta, Akash and Panda, Rameswar and Hyder, Rakib and Asif, M. Salman and Roy-Chowdhury, Amit K.},
title = {Non-Adversarial Video Synthesis With Learned Priors},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
pages = {6090--6099},
month = {June},
year = {2020}
}
```

## Contact 
Please contact the first author of the associated paper - Abhishek Aich (aaich001@ucr.edu) for any further queries.

