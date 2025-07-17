#!/home/guanyi/.local/bin/zsh

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49951  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit_daclip/deiqt_spaq.yaml \
--data-path /media/hdd1/hzh/iqa-dataset/SPAQ \
--output results \
--tensorboard \
--tag full_spaq_deit_daclip_v2_diffv3_DIN \
--repeat \
--rnum 10