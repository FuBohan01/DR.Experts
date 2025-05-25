from .deiqt import build_deiqt
from .dformer import build_dformer
from .mamba_vision import build_mamba
from .starnet import build_star
from .clip_lora import build_clip_lora
from .deit import build_deit_large

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == "deiqt":
        model = build_deiqt(
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            pretrained=config.MODEL.VIT.PRETRAINED,
            pretrained_model_path=config.MODEL.VIT.PRETRAINED_MODEL_PATH,
            infer=config.MODEL.VIT.CROSS_VALID,
            infer_model_path=config.MODEL.VIT.CROSS_MODEL_PATH,
        )
    elif model_type == "dformer":
        model = build_dformer(
            dims=config.MODEL.DINET.DIMS,
            mlp_ratios=config.MODEL.DINET.MLP_RATIOS,
            depths=config.MODEL.DINET.DEPTHS,
            num_heads=config.MODEL.DINET.NUM_HEADS,
            windows=config.MODEL.DINET.WINDOWS,
            pretrained=config.MODEL.DINET.PRETRAINED,
            pretrained_model_path=config.MODEL.DINET.PRETRAINED_MODEL_PATH,
            infer=config.MODEL.DINET.CROSS_VALID,
            infer_model_path=config.MODEL.DINET.CROSS_MODEL_PATH,
        )
    elif model_type == "mamba":
        model = build_mamba(
            depths = config.MODEL.MAMBA.DEPTHS,
            num_heads = config.MODEL.MAMBA.NUM_HEADS,
            window_size = config.MODEL.MAMBA.WINDOW_SIZE,
            dim = config.MODEL.MAMBA.DIM,
            in_dim = config.MODEL.MAMBA.IN_DIM,
            mlp_ratio = config.MODEL.MAMBA.MLP_RATIO,
            resolution =  config.MODEL.MAMBA.RESOLUTION,
            drop_path_rate = config.MODEL.MAMBA.DROP_PATH,
            pretrained=config.MODEL.MAMBA.PRETRAINED,
            pretrained_model_path=config.MODEL.MAMBA.PRETRAINED_MODEL_PATH,
            infer=config.MODEL.MAMBA.CROSS_VALID,
            infer_model_path=config.MODEL.MAMBA.CROSS_MODEL_PATH,
            layer_scale=config.MODEL.MAMBA.LAYER_SCALE
        )
    elif model_type == "star":
        model = build_star(
            depths = config.MODEL.STAR.DEPTHS,
            dims = config.MODEL.STAR.DIMS,
            mlp_ratio = config.MODEL.STAR.MLP_RATIO,
            pretrained=config.MODEL.STAR.PRETRAINED,
            pretrained_model_path=config.MODEL.STAR.PRETRAINED_MODEL_PATH,
            infer=config.MODEL.STAR.CROSS_VALID,
            infer_model_path=config.MODEL.STAR.CROSS_MODEL_PATH
        )
    elif model_type == "clip_lora":
        model = build_clip_lora(
            encoder=config.MODEL.CLIP_LORA.ENCODER,
            position=config.MODEL.CLIP_LORA.POSITION,
            params=config.MODEL.CLIP_LORA.PARAMS,
            r=config.MODEL.CLIP_LORA.R,
            alpha=config.MODEL.CLIP_LORA.ALPHA,
            dropout_rate=config.MODEL.CLIP_LORA.DROPOUT_RATE,
            backbone=config.MODEL.CLIP_LORA.BACKBONE
        )
    elif model_type == "deit_large":
        model = build_deit_large(
            pretrained=True
        )
    # elif model_type == "deit_large":
    #     model = build_deit_large(
    #         pretrained=True
    #     )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
