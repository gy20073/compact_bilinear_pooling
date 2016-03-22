This code implements compact bilinear pooling in MatConvNet. We also provide the implementation of several other pooling methods, such as bilinear pooling, fisher encoding and traditional features from FC7. We also provide scripts to compare those pooling methods on 4 datasets: MIT Indoor, CUB, DTD and FMD. The program entry point is 'main.m'. Before running experiments, you need to download the datasets and pretrained models, as detailed in the following text. 

Dataset
You need to download the dataset, unzip it and put it to the right location. Taking the CUB dataset as an example: download and unzip it to 'data/cub' directory, make sure directories such as 'data/cub/images' exists (sometimes the unzipped files might be contained in another directory, make sure you have the right directory structure).

CUB dataset
	http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
MIT indoor dataset
	We used the recommended subset of the MIT Indoor dataset. One could download the dataset directly from the official website, we also provide the subset we used. Simply download and unzip and you're all set. If you plan to start from the original dataset, you have to do some pre-processing to be compatible with our code. 
	https://drive.google.com/file/d/0B0ldy05JZFSacmpPaTNrN0NSUjA/view?usp=sharing
FMD dataset
	http://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip
DTD dataset 
	http://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz


ImageNet Pretrained Model
For the imagenet pretrained models, please download from the below urls to 'data/models', with the original filenames unchanged. N
http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-m.mat
http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat

All codes are tested on Ubuntu 14.04 and matlab 2016a. If you have any further questions or comments, please contact Yang Gao (yg@eecs.berkeley.edu). 

If you found our method useful, please consider citing our work:
@inproceedings{gao2016compact,
  title={Compact Bilinear Pooling},
  author={Gao, Yang and Beijbom, Oscar and Zhang, Ning and Darrell, Trevor},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on},
  year={2016}
}
