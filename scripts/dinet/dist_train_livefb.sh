#!/home/guanyi/.local/bin/zsh
# NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1,2,6,7 OMP_NUM_THREADS=4 torchrun --nnodes 1 --nproc_per_node 5 --master_port 49959  main.py \
--cfg configs/dinet/dinet_livefb.yaml \
--data-path /mnt/iMVR/guanyi/dataset/IQA/FLIVE_Database \
--output results \
--tensorboard \
--tag full_livefb_gaborfpn_new \
--repeat \
--rnum 10