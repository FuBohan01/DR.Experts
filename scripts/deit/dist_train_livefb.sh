#!/home/guanyi/.local/bin/zsh
# NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 torchrun --nnodes 1 --nproc_per_node 8 --master_port 49960  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit_daclip/deiqt_livefb.yaml \
--data-path /media/hdd1/hzh/iqa-dataset/liveFB \
--output /media/hdd1/fubohan/results \
--tensorboard \
--tag full_livefb_deit_daclip_v3_diffv5 \
--repeat \
--rnum 40