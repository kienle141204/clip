#!/bin/bash

python main.py \
    --model clip \
    --batch_size 64 \
    --num_epochs 30 \
    --lr 1e-4 \
    --embed_dim 256 \
    --checkpoint_dir checkpoints/clip
