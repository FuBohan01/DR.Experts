#!/home/guanyi/.local/bin/zsh

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=4 torchrun --nnodes 1 --nproc_per_node 2 --master_port 49959  main.py \
--cfg configs/dinet/dinet_kadid.yaml \
--data-path /mnt/iMVR/guanyi/dataset/IQA/kadid10k \
--output results \
--tensorboard \
--tag dinet_kadid_refo_fixed_LVFP3TP15P224N \
--repeat \
--rnum 10