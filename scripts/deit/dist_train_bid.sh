#!/home/guanyi/.local/bin/zsh
# NCCL_P2P_DISABLE=1 
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=4 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49959  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit_daclip/dinet_bid.yaml \
--data-path /media/hdd1/fubohan/Dataset/IQA/BID/ImageDatabase \
--output results \
--tensorboard \
--tag full_bid_deit_daclip_v2_diffv3_DIN_1 \
--repeat \
--rnum 40