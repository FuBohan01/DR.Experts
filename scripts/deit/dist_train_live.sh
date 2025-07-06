#!/home/guanyi/.local/bin/zsh

CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=4 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49958  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit_daclip/deiqt_live.yaml \
--data-path /media/hdd1/hzh/iqa-dataset/live/databaserelease2 \
--output results \
--tensorboard \
--tag full_live_deit[base]_daclip_v2_diffv3_DIN \
--repeat \
--rnum 10