#!/home/guanyi/.local/bin/zsh

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49945  main.py \
--cfg configs/vit/deiqt_koniq.yaml \
--data-path /mnt/iMVR/guanyi/dataset/IQA/KonIQ \
--output /mnt/iMVR/guanyi/dataset/IQA/results \
--tensorboard \
--tag diqa_orders_addmean_alternet_k10k \
--repeat \
--rnum 10