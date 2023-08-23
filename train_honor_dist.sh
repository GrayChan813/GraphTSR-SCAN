#! /bin/bash
filename="checkpoints/Log/honor.txt"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py \
    --dataroot '/data/cjc/Honor/' \
    --dataset_mode GBTSR_fromjson \
    --model tbrec \
    --batch_size 1 \
    --dataset_name Honor \
    --name Honor_tmp \
    --lr 0.0001 \
    --niter 51 \
    --niter_decay 49 \
    --print_freq 50 \
    --save_epoch_freq 5 \
    --save_latest_freq 108 \
    --no_html \
    --num_threads 16 \
    --epoch 35 \
    --epoch_count 36 \
    --rm_layers row_cls,col_cls \
    --load_height 800 \
    --load_width 800 \
    --continue_train \
    2>&1 | tee $filename
