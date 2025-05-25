#!/home/guanyi/.local/bin/zsh

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49942  main.py \
--cfg configs/vit/deiqt_livec.yaml \
--data-path /mnt/iMVR/guanyi/dataset/IQA/ChallengeDB_release \
--output /mnt/iMVR/guanyi/dataset/IQA/results \
--tensorboard \
--tag diqa_orders_addmean_alternet_newsel \
--repeat \
--rnum 10