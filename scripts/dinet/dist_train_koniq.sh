#!/home/guanyi/.local/bin/zsh
# NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=3,4,5 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 3 --master_port 49949  main.py \
--cfg configs/dinet/dinet_koniq.yaml \
--data-path /mnt/iMVR/guanyi/dataset/IQA/KonIQ \
--output results \
--tensorboard \
--tag full_k10k_gaborfpn_newsel \
--repeat \
--rnum 10