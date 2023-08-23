#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --dataroot ./datasets/ICDAR2013/test \
    --dataset_name ICDAR2013 \
    --dataset_mode merge_scitsrtable \
    --model tbrec \
    --batch_size 1 \
    --name ICDAR2013 \
    --num_rows 58 \
    --num_cols 13 \
    --num_threads 0 \
    --epoch 50 \
    --results_dir checkpoints/ICDAR2013/Split_Res_50 \
    --load_height 560 \
    --load_width 560