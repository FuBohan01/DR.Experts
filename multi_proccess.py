import os
 
# os.system("tmux attach -t aaai")
# 传递两个及以上参数
# LIVEC
os.system("CUDA_VISIBLE_DEVICES=0,3,5,7 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49949  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit_daclip/deit_livec.yaml \
--data-path /media/hdd1/hzh/iqa-dataset/livec \
--output /media/hdd1/fubohan/results \
--tensorboard \
--tag full_livec_deit_daclip_v3_diffv5_final_version \
--repeat \
--rnum 40")

# KONIQ-10K
os.system("CUDA_VISIBLE_DEVICES=0,3,5,7 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49948  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit_daclip/deit_koniq.yaml \
--data-path /home/fubohan/Dataset/koniq-10k \
--output /media/hdd1/fubohan/results \
--tensorboard \
--tag full_k10k_deit_daclip_v3_diffv5_final_version \
--repeat \
--rnum 40")

# SPAQ
os.system("CUDA_VISIBLE_DEVICES=0,3,5,7 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49951  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit_daclip/deiqt_spaq.yaml \
--data-path /media/hdd1/hzh/iqa-dataset/SPAQ \
--output /media/hdd1/fubohan/results \
--tensorboard \
--tag full_spaq_deit_daclip_v3_diffv5_final_version \
--repeat \
--rnum 40")

# os.system("CUDA_VISIBLE_DEVICES=2,5,6,7 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49948  main.py \
# --cfg /home/fubohan/Code/DIQA/configs/deit_daclip/deit_koniq.yaml \
# --data-path /home/fubohan/Dataset/koniq-10k \
# --output /media/hdd1/fubohan/results \
# --tensorboard \
# --tag full_k10k_deit_daclip_v3_diffv5_din_bestversion_occuipy \
# --repeat \
# --rnum 100")