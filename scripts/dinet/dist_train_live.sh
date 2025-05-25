#!/home/guanyi/.local/bin/zsh

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 torchrun --nnodes 1 --nproc_per_node 1 --master_port 49952  main.py \
--cfg configs/dinet/dinet_live.yaml \
--data-path /mnt/iMVR/guanyi/dataset/IQA/databaserelease2 \
--output results \
--tensorboard \
--tag dinet_live_save_ckpts \
--repeat \
--rnum 10