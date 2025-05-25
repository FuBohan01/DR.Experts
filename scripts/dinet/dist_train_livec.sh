#!/home/guanyi/.local/bin/zsh
# NCCL_P2P_DISABLE=1 
CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 2 --master_port 50001  main.py \
--cfg configs/dinet/dinet_livec.yaml \
--data-path /mnt/iMVR/guanyi/dataset/IQA/ChallengeDB_release \
--output results \
--tensorboard \
--tag full_livec_gaborfpn \
--repeat \
--rnum 10