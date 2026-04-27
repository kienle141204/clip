#!/bin/bash

nohup python -u main.py \
    --model clip \
    --batch_size 64 \
    --num_epochs 50 \
    --lr 1e-4 \
    --embed_dim 256 \
    --checkpoint_dir checkpoints/clip \
    > logs/clip.log 2>&1 &

echo "Training started. PID: $!"
echo "Monitor logs: tail -f logs/clip.log"
