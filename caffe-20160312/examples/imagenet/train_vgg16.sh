#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/vgg16/solver.prototxt --gpu=4
