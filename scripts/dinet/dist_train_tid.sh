#!/home/guanyi/.local/bin/zsh

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=4 torchrun --nnodes 1 --nproc_per_node 2 --master_port 49958  main.py \
--cfg configs/dinet/dinet_tid.yaml \
--data-path /mnt/iMVR/guanyi/dataset/IQA/tid2013 \
--output results \
--tensorboard \
--tag dinet_tid_refo_fixed_LVFP5TP15P224N \
--repeat \
--rnum 10