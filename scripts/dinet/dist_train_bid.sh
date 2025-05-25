#!/home/guanyi/.local/bin/zsh
# NCCL_P2P_DISABLE=1 
CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=4 torchrun --nnodes 1 --nproc_per_node 2 --master_port 49959  main.py \
--cfg configs/dinet/dinet_bid.yaml \
--data-path /mnt/iMVR/guanyi/dataset/IQA/BID/ImageDatabase \
--output results \
--tensorboard \
--tag dinet_bid_refo_fixed_LVFP8TP15P224S \
--repeat \
--rnum 10