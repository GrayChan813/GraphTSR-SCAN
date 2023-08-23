#! /bin/bash
filename="checkpoints/Log/scitsr.txt"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py \
    --dataroot '/data/cjc/SciTSR/train' \
    --dataset_mode GBTSR_fromjson \
    --model tbrec \
    --batch_size 1 \
    --dataset_name SciTSR \
    --name scitsr_tmp \
    --lr 0.0001 \
    --niter 51 \
    --niter_decay 49 \
    --print_freq 50 \
    --save_epoch_freq 5 \
    --no_html \
    --num_threads 16 \
    --epoch 0 \
    --epoch_count 1 \
    --load_height 800 \
    --load_width 800 \
    2>&1 | tee $filename
