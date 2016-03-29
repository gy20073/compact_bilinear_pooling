This is the bilinear pooling and compact bilinear pooling caffe implementation.
The original implementation is in MatConvNet and for convenience, we port them to Caffe. 

Sample layer prototxt: \
Bilinear layer:

    layer {
      name: "bilinear_layer"
      type: "Bilinear"
      bottom: "in1"
      bottom: "in2"
      top: "out"
    }

compact bilinear Tensor Sketch layer:

    layer {
      name: "compact_bilinear"
      type: "CompactBilinear"
      bottom: "in1"
      bottom: "in2"
      top: "out"
      compact_bilinear_param {
        num_output: 4096
        sum_pool: false
      }
    }

We only implemented the compact bilinear Tensor Sketch version without learning the random weights, since it's the best in practice. For convenience, we also implement the signed-sqrt layer and the (sample-wise) l2 normalization layer, as:
    
    layer {
      name: "signed_sqrt_layer"
      type: "SignedSqrt"
      bottom: "in"
      top: "out"
    }
and

    layer {
      name: "l2_normalization_layer"
      type: "L2Normalization"
      bottom: "in"
      top: "out"
    }
the usual use cases are compact_bilinear + signed-sqrt + l2_normalization + classification. 


For both bilinear and compact bilinear layer, two inputs could be the same blob,
i.e. in1==in2. But we always require two inputs. The two input sizes must be compatible with each other. "in1" and "in2" should have shapes: N\*C1\*H\*W and N\*C2\*H\*W respectively. Only the number of channels could be different. 

The bilinear layer always output a blob with a shape of N\*(C1\*C2)\*1\*1, i.e. bilinear features 
that is spatially sum pooled. The compact bilinear layer's output shape, on the other hand, depend on its compact_bilinear_param. In addition to the spatially sum pooled feature (output size N\*num\_output\*1\*1, we also allow the non-pooled feature (sum_pool: false, output size\: N\*num\_output\*H\*W). This could be useful in the case where one needs some spatio resolution in the output, such as keypoint detection.


TODO list:
1. make an example network prototxt on CUB dataset. 


If you find bilinear pooling or compact bilinear pooling, please consider citing:

    @inproceedings{lin2015bilinear,
      title={Bilinear CNN models for fine-grained visual recognition},
      author={Lin, Tsung-Yu and RoyChowdhury, Aruni and Maji, Subhransu},
      booktitle={Proceedings of the IEEE International Conference on Computer Vision},
      pages={1449--1457},
      year={2015}
    }
and

    @inproceedings{gao2016compact,
      title={Compact Bilinear Pooling},
      author={Gao, Yang and Beijbom, Oscar and Zhang, Ning and Darrell, Trevor},
      booktitle={Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on},
      year={2016}
    }





## Original Caffe Readme

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
