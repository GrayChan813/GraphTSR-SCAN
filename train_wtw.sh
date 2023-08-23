#! /bin/bash
filename="checkpoints/WTW/log.txt"
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --dataroot '/data/cjc/WTW/train/' \
    --dataset_mode GBTSR_fromjson \
    --model tbrec \
    --batch_size 1 \
    --dataset_name WTW \
    --name WTW_tmp \
    --lr 0.0001 \
    --niter 41 \
    --niter_decay 39 \
    --print_freq 50 \
    --save_epoch_freq 5 \
    --save_latest_freq 108 \
    --no_html \
    --num_threads 16 \
    --epoch 0 \
    --epoch_count 0 \
    --rm_layers row_cls,col_cls \
    --load_height 800 \
    --load_width 800 \
    2>&1 | tee $filename
