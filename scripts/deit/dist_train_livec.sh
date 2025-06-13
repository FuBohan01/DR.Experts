#!/home/guanyi/.local/bin/zsh
# NCCL_P2P_DISABLE=1 
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=4,5,6 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 3 --master_port 49949  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit/deit_livec.yaml \
--data-path /media/hdd1/hzh/iqa-dataset/livec \
--output results \
--tensorboard \
--tag full_livec_daclip_more_norm \
--repeat \
--rnum 5

CUDA_VISIBLE_DEVICES=1,2,3 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 3 --master_port 49949  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit/deit_livec.yaml \
--data-path /media/hdd1/hzh/iqa-dataset/livec \
--output results \
--tensorboard \
--tag full_livec_deit_daclip_v2_diffv3_multikanlayer \
--repeat \
--rnum 10