#!/bin/bash

mkdir -p log

for data in CUB_200_2011 Stanford_Cars FGVC_Aircraft; do
  CUDA_VISIBLE_DEVICES=0 python -u finetune.py --max_iters=9000 --task_name=$data --data_dir=./benchmark/$data --regularizer=smile 2>&1 1>log/log.$data
done
