#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --dataroot '/data/cjc/WTW/test/' \
    --dataset_mode GBTSR_fromjson \
    --model tbrec \
    --batch_size 1 \
    --dataset_name WTW \
    --name WTW_tmp \
    --num_threads 0 \
    --epoch latest \
    --results_dir checkpoints/SciTSR/Split_Res_50 \
    --load_height 600 \
    --load_width 600
