import os
from PIL import Image

from visualizer import get_local

get_local.activate()

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from models.dformer import build_dformer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def process_inverse_depth(depth):
    depth = (255 - depth) / 255.0

    mean = 0.485
    std = 0.229

    depth = (depth - mean) / std

    return depth


model = build_dformer(
    dims=[96, 192, 288, 576],
    mlp_ratios=[8, 8, 4, 4],
    depths=[3, 3, 12, 2],
    num_heads=[1, 2, 4, 8],
    windows=[0, 7, 7, 7],
    infer=True,
    infer_model_path="results/ablation-k10k/full_k10k_gaborfpn/0/ckpt_epoch_8.pth",
).cuda()

transform = [
    transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.Resize((224, 224)),
        ]
    ),
    transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
    transforms.Compose(
        [
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    ),
]
img_path = "/mnt/iMVR/guanyi/dataset/IQA/KonIQ/5980068668.jpg"
directory, filename = os.path.split(img_path)
name, ext = os.path.splitext(filename)

depth_dir = directory.replace("KonIQ", "KonIQ_Depths")
depths_file = f"{name}-depth.png"
orders_file = f"{name}-onehot_depth.npy"

depths_file = os.path.join(depth_dir, depths_file)
orders_file = os.path.join(depth_dir, orders_file)
vis_img = Image.open(img_path)

sample = np.array(Image.open(img_path)).astype(np.float32)
cor_depth, cor_orders = (
    np.array(Image.open(depths_file).convert("L")).astype(np.float32),
    np.load(orders_file, allow_pickle=True).astype(np.float32),
)
sample = torch.from_numpy(
    np.transpose(
        np.concatenate([sample, cor_depth[:, :, None], cor_orders], axis=-1),
        (2, 0, 1),
    )
).contiguous()
sample = transform[0](sample)
images, depths, orders = (
    (sample[:3, :, :] / 255.0).contiguous(),
    sample[3, :, :].contiguous(),
    sample[4:, :, :].contiguous().cuda(),
)
depths = process_inverse_depth(depths).cuda()
images = transform[2](images).unsqueeze(0).cuda()
with torch.no_grad():
    output = model(images, depths.unsqueeze(0), [orders])
print(output)


def gray2rgb(image):
    return np.repeat(image[..., np.newaxis], 3, 2)


def visualize_grid_to_grid(att_map, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    H, W = att_map.shape

    grid_image = highlight_grid(image, grid_size)

    mask = att_map.reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.tight_layout()

    # ax[0].imshow(grid_image)
    # ax[0].axis("off")

    ax.imshow(grid_image)
    ax.imshow(mask / np.max(mask), alpha=alpha, cmap="rainbow")
    ax.axis("off")
    plt.savefig("assets/heatmap.png", bbox_inches="tight", pad_inches=0)


def highlight_grid(image, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    W, H = image.size
    return image


cache = get_local.cache
attention_maps = cache["ModifiedTransformerDecoderLayer._mha_block"][0]
attention_maps = attention_maps[0, 0, :49].reshape(7, 7)
visualize_grid_to_grid(attention_maps, vis_img, grid_size=7)
