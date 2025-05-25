#!/home/guanyi/.local/bin/zsh

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 2 --master_port 49950  main.py \
--cfg configs/dinet/dinet_spaq.yaml \
--data-path /mnt/iMVR/guanyi/dataset/IQA/SPAQ \
--output results \
--tensorboard \
--tag dinet_spaq_refo_fixed_LVFP3TP15P224N \
--repeat \
--rnum 10