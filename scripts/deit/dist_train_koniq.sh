#!/home/guanyi/.local/bin/zsh
# NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49948  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit_daclip/deit_koniq.yaml \
--data-path /home/fubohan/Dataset/koniq-10k \
--output results \
--tensorboard \
--tag full_k10_deit[base]_daclip_v2_diffv3_DIN \
--repeat \
--rnum 10