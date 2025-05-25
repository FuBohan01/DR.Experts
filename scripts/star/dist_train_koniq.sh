#!/home/guanyi/.local/bin/zsh
# NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=0,4,5 OMP_NUM_THREADS=1 torchrun --nnodes 1 --nproc_per_node 3 --master_port 49956  main.py \
--cfg /home/fubohan/Code/DIQA-dev/configs/star/star_koniq.yaml \
--data-path /home/fubohan/Dataset/koniq-10k \
--output results \
--tensorboard \
--tag star_test_for_backbone \
--repeat \
--rnum 10