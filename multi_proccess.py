import os
 
# os.system("tmux attach -t aaai")
# 传递两个及以上参数
os.system("CUDA_VISIBLE_DEVICES=2,5,6,7 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49949  main.py \
--cfg /home/fubohan/Code/DIQA/configs/deit_daclip/deit_livec.yaml \
--data-path /media/hdd1/hzh/iqa-dataset/livec \
--output /media/hdd1/fubohan/results \
--tensorboard \
--tag full_livec_deit_daclip_v3_diffv5_din_bestversion_class[noise_rainy_hazy_jpeg_shadow] \
--repeat \
--rnum 40")


# os.system("CUDA_VISIBLE_DEVICES=2,5,6,7 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49948  main.py \
# --cfg /home/fubohan/Code/DIQA/configs/deit_daclip/deit_koniq.yaml \
# --data-path /home/fubohan/Dataset/koniq-10k \
# --output /media/hdd1/fubohan/results \
# --tensorboard \
# --tag full_k10k_deit_daclip_v3_diffv5_din_bestversion_class[uncompleted] \
# --repeat \
# --rnum 40")

# os.system("CUDA_VISIBLE_DEVICES=2,5,6,7 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49948  main.py \
# --cfg /home/fubohan/Code/DIQA/configs/deit_daclip/deit_koniq.yaml \
# --data-path /home/fubohan/Dataset/koniq-10k \
# --output /media/hdd1/fubohan/results \
# --tensorboard \
# --tag full_k10k_deit_daclip_v3_diffv5_din_bestversion_occuipy \
# --repeat \
# --rnum 100")