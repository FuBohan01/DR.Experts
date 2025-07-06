#!/home/guanyi/.local/bin/zsh

CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=4 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49952  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit/deiqt_csiq.yaml \
--data-path /media/hdd1/fubohan/Dataset/IQA/CSIQ \
--output results \
--tensorboard \
--tag full_csiq_deit_daclip_v2_diffv3_DIN \
--repeat \
--rnum 10