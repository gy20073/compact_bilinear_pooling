#!/bin/bash

# first fine all layers

./build/tools/caffe train \
	-model "examples/compact_bilinear/ft_all.prototxt" \
	-solver "examples/compact_bilinear/ft_all.solver" \
	-weights "examples/compact_bilinear/snapshot/ft_last_layer_iter_60000.caffemodel" \
	-gpu 0
	