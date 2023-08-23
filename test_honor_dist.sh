#! /bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 test.py \
    --dataroot '/data/cjc/Honor/' \
    --dataset_mode GBTSR_fromjson \
    --model tbrec \
    --batch_size 1 \
    --dataset_name Honor \
    --name Honor_tmp \
    --num_threads 16 \
    --epoch 100 \
    --results_dir checkpoints/SciTSR/Split_Res_50 \
    --load_height 800 \
    --load_width 800
