CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 2 --master_port 49958  valid.py \
--cfg /home/fubohan/Code/DIQA/configs/cross_daclip/cross_to_koniq.yaml \
--data-path /home/fubohan/Dataset/koniq-10k \
--output /media/hdd1/fubohan/results \
--tag livec7_to_koniq_[7][05] \
--repeat \
--rnum 10

CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 2 --master_port 49958  valid.py \
--cfg /home/fubohan/Code/DIQA/configs/cross_daclip/cross_to_livec.yaml \
--data-path /media/hdd1/hzh/iqa-dataset/livec \
--output /media/hdd1/fubohan/results \
--tag livefb_to_livec[7][08] \
--repeat \
--rnum 10

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 1 --master_port 49958  valid.py \
--cfg configs/cross/cross_to_csiq.yaml \
--data-path /mnt/iMVR/guanyi/dataset/IQA/CSIQ \
--output /media/hdd1/fubohan/results \
--tag diqa_live_to_csiq \
--repeat \
--rnum 10