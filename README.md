# Non-Adversarial Video Synthesis with Learned Priors

## Overview
This package is a PyTorch implementation of the paper "Non-Adversarial Video Synthesis</a> with Learned Priors" accepted at [IEEE CVPR 2020](http://cvpr2020.thecvf.com/).</br> [[Project page]](https://abhishekaich27.github.io/data/Project_pages/CVPR_2020/navsynth.html)

## Dependencies
Create a conda environment with the 'environment.yml'

## Data
Please download the Chair-CAD dataset 'https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar'.

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
@misc{aich2020nonadversarial,
    title={Non-Adversarial Video Synthesis with Learned Priors},
    author={Abhishek Aich and Akash Gupta and Rameswar Panda and Rakib Hyder and M. Salman Asif and Amit K. Roy-Chowdhury},
    year={2020},
    eprint={2003.09565},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```

## Contact 
Please contact the first author of the associated paper - Abhishek Aich (aaich001@ucr.edu) for any further queries.

