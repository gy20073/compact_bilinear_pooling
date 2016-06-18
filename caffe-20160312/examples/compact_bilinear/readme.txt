Usage:

1. download the VGG16 models into the current directory: 
	$CAFFE_ROOT/examples/compact_bilinear/VGG_ILSVRC_16_layers.caffemodel
	Model URL: https://gist.github.com/ksimonyan/211839e770f7b538e2d8

2. download the CUB dataset at: http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz
	Unzip it to current folder. Please make sure that the images are availble at the locations like: $CAFFE_ROOT/examples/compact_bilinear/cub/images/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg

3. fine tune the last layer (assume you're at the $CAFFE_ROOT)
	./examples/compact_bilinear/ft_last_layer.sh

	This step is equivalent to train a logistic regression classifier. Practically, it's faster to extract features from images and train that using any offline solver, as what we did in the MatConvNet version. But since it requires caching features and call a third party tool, it's often messier than this approach. 

4. fine tune the whole network
	./examples/compact_bilinear/ft_all.sh

5. The fine tuned models are available at:
	https://drive.google.com/open?id=0B0ldy05JZFSaS1pIUjBaSUxMTUU

Trained on cub200 with random cropping and achieve an accuracy of 84.98% 