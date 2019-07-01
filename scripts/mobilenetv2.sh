#!/usr/bin/env bash

# train
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model mobilenetv2 \
    --base-size 480 \
    --crop-size 480 \
    --workers 12 \
    --batch-size 8 \
    --dataset segdata \
    --lr 1e-2 \
    --log-iter 100 \
    --save-epoch 100 \
    --epochs 200

# # eval
# CUDA_VISIBLE_DEVICES=1 python eval.py --model mobilenetv2 \
#     --dataset citys --aux

# # fps
# CUDA_VISIBLE_DEVICES=1 python test_fps.py --model mobilenetv2 \
#     --dataset citys --aux
