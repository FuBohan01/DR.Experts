import argparse
import datetime
import json
import logging
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
import torchmetrics
from timm.utils import AverageMeter  # accuracy
from torchinfo import summary

from config import get_config
from IQA import IQA_build_loader
from logger import create_logger
from lr_scheduler import build_scheduler
from models import build_model
from optimizer import build_optimizer
from utils import (
    NativeScalerWithGradNormCount,
    auto_resume_helper,
    load_checkpoint,
    load_pretrained,
    reduce_tensor,
    save_checkpoint,
    plcc_loss,
)
from scipy import io
from PIL import Image
from torchvision import transforms
from models.deit import build_deit_large


with open("/home/fubohan/Code/DIQA/results/deiqt_small/full_livec_daclip_multihead[test1]/8/sel_num.data", "rb") as f:
    sel_num = pickle.load(f)

train_index = sel_num[0 : int(round(0.8 * len(sel_num)))]
test_index = sel_num[int(round(0.8 * len(sel_num))) : len(sel_num)]

root = "/media/hdd1/hzh/iqa-dataset/livec"

imgpath = io.loadmat(os.path.join(root, "Data", "AllImages_release.mat"))
imgpath = imgpath["AllImages_release"]
imgpath = imgpath[7:1169]
mos = io.loadmat(os.path.join(root, "Data", "AllMOS_release.mat"))
labels = mos["AllMOS_release"].astype(np.float32)
labels = labels[0][7:1169]

sample = []
for i, item in enumerate(test_index):
    for aug in range(15):
        sample.append(
            (os.path.join(root, "Images", imgpath[item][0][0]), labels[item])
        )

def load_image(path, mode="RGB"):
    try:
        im = Image.open(path).convert(mode)
    except:
        print("ERROR IMG LOADED: ", path)
        random_img = np.random.rand(224, 224, 3) * 255
        im = Image.fromarray(np.uint8(random_img))
    return im

transform = [
    transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop(size=(384, 384)),
        ]
    ),
    transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
    transforms.Compose(
        [
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    ),
]
output_file_path = "/home/fubohan/Code/DIQA/output_scores.txt"
with open(output_file_path, "w") as f:
    for i in range(len(sample)):

        path, target = sample[i]

        x = load_image(path)

        x = transform[0](x)
        x = transform[1](x)
        x = transform[2](x)

        model = build_deit_large().cuda()  # 确保模型在GPU上
        model.load_state_dict(
            torch.load(
                "/home/fubohan/Code/DIQA/results/deiqt_small/full_livec_daclip_more_norm/8/ckpt_epoch_8.pth"
            )["model"])
        x = x.unsqueeze(0).cuda()  # 添加batch维度并移动到GPU
        y = model(x).cuda()
        y = y.item() if isinstance(y, torch.Tensor) else y  # 转换为标量

        # 将信息写入文件
        f.write(f"{path} {y} {target}\n")
