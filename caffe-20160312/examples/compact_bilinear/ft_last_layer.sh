#!/bin/bash

# first fine tune the last layer only

./build/tools/caffe train \
	-model "examples/compact_bilinear/ft_last_layer.prototxt" \
	-solver "examples/compact_bilinear/ft_last_layer.solver" \
	-weights "examples/compact_bilinear/VGG_ILSVRC_16_layers.caffemodel" \
	-gpu 0
	