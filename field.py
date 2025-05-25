import torch

checkpoint = torch.load(
    "results/ablation-livec/dinet_livec_save_ckpts/0/ckpt_epoch_0.pth",
    map_location="cpu",
    weights_only=False,
)
pass
