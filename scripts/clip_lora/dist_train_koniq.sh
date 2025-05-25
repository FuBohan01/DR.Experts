#!/home/guanyi/.local/bin/zsh
# NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49949  main.py \
--cfg /home/fubohan/Code/DIQA-dev/configs/clip_lora/clip_lora_koniq.yaml \
--data-path /home/fubohan/Dataset/koniq-10k \
--output results \
--tensorboard \
--tag full_k10_lr5-4-3_vitlarge_Ploss_hiratrain \
--repeat \
--rnum 5