#!/home/guanyi/.local/bin/zsh

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 torchrun --nnodes 1 --nproc_per_node 2 --master_port 49958  main.py \
--cfg configs/dinet/dinet_csiq.yaml \
--data-path /mnt/iMVR/guanyi/dataset/IQA/CSIQ \
--output results \
--tensorboard \
--tag dinet_csiq_save_ckpts \
--repeat \
--rnum 10