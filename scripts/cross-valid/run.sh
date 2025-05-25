CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 1 --master_port 49958  valid.py \
--cfg configs/cross/cross_to_koniq.yaml \
--data-path /mnt/iMVR/guanyi/dataset/IQA/KonIQ \
--output results \
--tag diqa_livefb_to_koniq \
--repeat \
--rnum 10

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 1 --master_port 49958  valid.py \
--cfg configs/cross/cross_to_livec.yaml \
--data-path /mnt/iMVR/guanyi/dataset/IQA/ChallengeDB_release \
--output results \
--tag diqa_livefb_to_livec \
--repeat \
--rnum 10

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 1 --master_port 49958  valid.py \
--cfg configs/cross/cross_to_csiq.yaml \
--data-path /mnt/iMVR/guanyi/dataset/IQA/CSIQ \
--output results \
--tag diqa_live_to_csiq \
--repeat \
--rnum 10