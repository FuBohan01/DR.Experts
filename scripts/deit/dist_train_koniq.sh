#!/home/guanyi/.local/bin/zsh
# NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=2,5,6,7 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49948  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit_daclip/deit_koniq.yaml \
--data-path /home/fubohan/Dataset/koniq-10k \
--output /media/hdd1/fubohan/results \
--tensorboard \
--tag full_k10k_deit_daclip_v3_diffv5_din_bestversion_occuipy \
--repeat \
--rnum 100

CUDA_VISIBLE_DEVICES=2,5,6,7 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49948  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit_daclip/deit_koniq.yaml \
--data-path /home/fubohan/Dataset/koniq-10k \
--output /media/hdd1/fubohan/results \
--tensorboard \
--tag full_k10k_deit_daclip_v3_diffv5_din_bestversion_model[base] \
--repeat \
--rnum 40