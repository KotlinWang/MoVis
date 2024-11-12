#!/bin/bash

count=2

while [ $count -lt 10 ]; do
    CUDA_VISIBLE_DEVICES='5' python tools/train_val.py --config $1 --checkpoint_name $2"_$((count + 1))"  > logs/$(date +"%Y%m%d_%H:%M").log
    # 释放显存
    # nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r -I{} kill {}
    # # 终止Python脚本进程
    # pkill -f "python tools/train_val.py --config $1 --checkpoint_name $2"_"$((count + 1))"
    # 休眠1分钟
    sleep 10
    count=$((count + 1))
done

# CUDA_VISIBLE_DEVICES='5' python tools/train_val.py --config $1 --checkpoint_name $2"_$((count + 1))"  > logs/$(date +"%Y%m%d_%H:%M").log