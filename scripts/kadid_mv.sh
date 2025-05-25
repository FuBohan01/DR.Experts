#!/bin/bash

# 设置 Kadid 数据集根目录
KADID_DIR="/mnt/iMVR/guanyi/dataset/IQA/kadid10k/images"
# 目标存放参考图像的目录
REFERENCE_DIR="/mnt/iMVR/guanyi/dataset/IQA/kadid10k/reference_images"

# 确保目标目录存在
mkdir -p "$REFERENCE_DIR"

# 查找所有符合 Ixx.png 格式的文件（即没有 _yy_zz 尾注）
find "$KADID_DIR" -type f -regextype posix-egrep -regex ".*/I[0-9]{2}\.png$" -exec mv {} "$REFERENCE_DIR" \;

echo "所有参考图像已移动到 $REFERENCE_DIR"