#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/vggm/solver.prototxt --gpu=4
