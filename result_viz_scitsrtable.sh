#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python result_vis.py \
    --dataroot '/data/cjc/WTW/test/' \
    --dataset_mode GBTSR_fromjson \
    --model tbrec \
    --batch_size 1 \
    --dataset_name WTW \
    --name WTW_tmp \
    --num_threads 0 \
    --epoch 100 \
    --results_dir checkpoints/SciTSR/Split_Res_50 \
    --load_height 800 \
    --load_width 800 \
