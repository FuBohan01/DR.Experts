#!/home/guanyi/.local/bin/zsh
# NCCL_P2P_DISABLE=1 
CUDA_VISIBLE_DEVICES=0,3,5,7 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49949  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit_daclip/deit_livec.yaml \
--data-path /media/hdd1/hzh/iqa-dataset/livec \
--output /media/hdd1/fubohan/results \
--tensorboard \
--tag full_livec_deit_daclip_v3_diffv5_final_version \
--repeat \
--rnum 40

CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49949  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit_daclip/deit_livec.yaml \
--data-path /media/hdd1/hzh/iqa-dataset/livec \
--output results \
--tensorboard \
--tag full_livec_deit_daclip_v2_diffv3_ comprehensive_feature \
--repeat \
--rnum 10