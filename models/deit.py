# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from . import open_clip
# import open_clip
from math import sqrt
import pdb

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 
    
class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Layer_scale_init_Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_1_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        
    def forward(self, x):
        x = x + self.drop_path(self.gamma_1*self.attn(self.norm1(x))) + self.drop_path(self.gamma_1_1 * self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) + self.drop_path(self.gamma_2_1 * self.mlp1(self.norm21(x)))
        return x
        
class Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.mlp1(self.norm21(x)))
        return x
        
        
class hMLP_stem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, embed_dim=768,norm_layer=nn.SyncBatchNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(*[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2),
                                          norm_layer(embed_dim),
                                         ])
        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = Block,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,
                dpr_constant=True,init_scale=1e-4,
                mlp_ratio_clstk = 4.0,**kwargs):
        super().__init__()
        
        self.dropout_rate = drop_rate

            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        

        
            
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = x + self.pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)
            
        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        x = self.norm(x)  # [1, 197, 384]
        # return x[:, 0]
        return x  # [bs, 196, 384] remove cls token

    def forward(self, x):

        x = self.forward_features(x)
        
        # if self.dropout_rate:
        #     x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        # x = self.head(x)
        
        return x

class diff_attention(nn.Module):
    def __init__(self, embedding_dim, alpha=0.5):
        super(diff_attention, self).__init__()
        self.W_q1 = nn.Linear(embedding_dim, embedding_dim)
        self.W_q2 = nn.Linear(embedding_dim, embedding_dim)
        self.W_k1 = nn.Linear(embedding_dim, embedding_dim)
        self.W_k2 = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim * 2, embedding_dim)  # Changed to output d dimensions
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        # self.alpha_fc = nn.Linear(embedding_dim * 2, 1)
        self.embedding_dim = embedding_dim
        # self.score_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

    def forward(self, X1, X2):
        """
        Forward pass of the Differential Attention module.
        
        Args:
        - X (Tensor): Input tensor.
        - λ (float): Scaling factor for the difference.
        
        Returns:
        - Tensor: Output tensor.
        """
        X1 = X1.unsqueeze(1)  # [bs, 1, 384]
        X1 = X1.expand(-1, X2.shape[1], -1)  # [bs, 576, 384]
        X = torch.cat([X1, X2], dim=-1)
        Q1 = self.W_q1(X1)
        K1 = self.W_k1(X1)
        Q2 = self.W_q2(X2)
        K2 = self.W_k2(X2)      
        V = self.W_v(X)

        s = 1 / sqrt(self.embedding_dim)
        
        A1 = (Q1 @ K1.transpose(-1, -2)) * s
        A2 = (Q2 @ K2.transpose(-1, -2)) * s
        
        A1_softmax = F.softmax(A1, dim=-1)
        A2_softmax = F.softmax(A2, dim=-1)
        
        # alpha = torch.relu(self.alpha_fc(X))
        # result = (A2_softmax *(1 - self.alpha * A1_softmax)) @ V   
        result = (A2_softmax - self.alpha * A1_softmax) @ V
        # return result[:, 0] #[bs, 586, 384]
        return result #[bs, 576, 384]


# DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)
class dascore_vit_models(nn.Module):
    def __init__(self, img_size=384):
        super().__init__()
        self.backbone = vit_models(img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                                norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block,num_classes=1)
        
        checkpoint = '/home/fubohan/Code/DIQA/checkpoint/daclip_ViT-B-32.pt'
        self.head, self.head_preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=checkpoint)
        degradations = ['motion-blurry','hazy','jpeg-compressed','low-light','noisy','raindrop','rainy','shadowed','snowy','uncompleted']

        self.L1 = nn.Linear(512, 384)
        self.L2 = nn.Linear(576, 1)  # 196 is the number of patches in ViT-B-32
        self.diff_attention = diff_attention(384)  # 384 is the embedding dimension of ViT-B-32
        self.score = nn.Linear(384, 1)

        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.text = tokenizer(degradations).cuda()
        self.lamda_init = nn.Parameter(torch.zeros(1, 384), requires_grad=True)

        self.attn_norm = nn.LayerNorm(384 * len(degradations))
        self.group_attn_ffn = nn.Sequential(
            nn.Linear(384 * len(degradations), 384 * len(degradations)),
            nn.GELU(),
            nn.Linear(384 * len(degradations), 384)
        )
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(384)
        self.norm3 = nn.LayerNorm(384)

    def mulithead(self, da_metrics, img_feature):
        group_attn = []
        for i in range(da_metrics.shape[1]):
            diff_attn = self.diff_attention(da_metrics[:, i, :], img_feature)  # [bs, 576, 384]
            
            group_attn.append(diff_attn)
        group_attn = torch.cat(group_attn, dim=-1)  # [bs, 576, num_group * C]
        # Norm 和 FFN
        group_attn = self.attn_norm(group_attn)  # [bs, 576, 384]
        # group_attn = self.attn_norm(group_attn)  # [bs, 576, C]
        group_attn = self.group_attn_ffn(group_attn)
        # 
        group_attn = group_attn * (1 - self.lamda_init)
        return group_attn
        

    def forward(self, x):
        # pdb.set_trace()
        img_feature = self.backbone(x) # [1, 576, 384]
        # da_img = 
        da_img = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=da_img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=da_img.device).view(1, 3, 1, 1)
        da_img = (da_img - mean) / std

        text_features = self.head.encode_text(self.text)
        clip_image_feature, da_img_feature = self.head.encode_image(da_img, control= True) # shape=[1, 512] 
        da_img_feature = da_img_feature / da_img_feature.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 先扩展维度
        text_features_expand = text_features.unsqueeze(0).expand(da_img_feature.shape[0], -1, -1)  # [52, nclass, 512]
        da_img_feature_expand = da_img_feature.unsqueeze(1)  # [52, 1, 512]

        # 按元素相乘
        da_metrics = text_features_expand * da_img_feature_expand  # [bs, nclass, 512]
        da_metrics = self.norm1(da_metrics)  # [52, n_class, 512]
        da_metrics = self.L1(da_metrics) # shape=[bs, n_class, 384]
        
        img_feature = self.norm2(img_feature)  # [52, 576, 384]

        out = self.mulithead(da_metrics, img_feature) # shape=[bs, 576, 384]
        out = out[:, 0, :]
        # out = out.permute(0, 2, 1)  # [52, 384, 576]
        # out = self.L2(out)  # [52, 384, 1]
        # out = out.squeeze(-1)  # [52, 384]
        # out = self.norm3(out)  # [52, 384]
        out = self.score(out)
        return out

def build_deit_large(pretrained=False, img_size=384, pretrained_21k=True, **kwargs):
    # 创建主模型
    model = dascore_vit_models(img_size=img_size)
    # 如果需要加载backbone的预训练权重
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_small_' + str(img_size) + '_'
        if pretrained_21k:
            name += '21k.pth'
        else:
            name += '1k.pth'
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        state_dict = checkpoint['model']
        # 删除head权重，避免shape不匹配
        if 'head.weight' in state_dict:
            del state_dict['head.weight']
        if 'head.bias' in state_dict:
            del state_dict['head.bias']
        # 加载到backbone
        model.backbone.load_state_dict(state_dict, strict=False)
        for name, param in model.named_parameters():
            if 'head' in name:
                param.requires_grad = False
            elif 'text' in name:
                param.requires_grad = False
    return model

# def build_deit_large(pretrained=False, img_size=384, pretrained_21k = True,  **kwargs):
#     model = vit_models(
#         img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block,num_classes=1, **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         name = 'https://dl.fbaipublicfiles.com/deit/deit_3_small_'+str(img_size)+'_'
#         if pretrained_21k:
#             name+='21k.pth'
#         else:
#             name+='1k.pth'
            
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url=name,
#             map_location="cpu", check_hash=True
#         )
#         state_dict = checkpoint['model']
#         del state_dict['head.weight']
#         del state_dict['head.bias']
#         model.load_state_dict(state_dict, strict=False)

#     return model

def deit_base_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_base_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
    
def deit_large_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
# def build_deit_large1(pretrained=False, img_size=224, pretrained_21k = True,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, num_classes=1, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_large_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        state_dict = checkpoint['model']
        del state_dict['head.weight']
        del state_dict['head.bias']
        model.load_state_dict(state_dict, strict=False)
    return model
    
if __name__ == '__main__':
    model = build_deit_large(img_size=384)
    # print(model)
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print(f"Frozen: {name} | shape: {param.shape}")
    #     else:
    #         print(f"Trainable: {name} | shape: {param.shape}")
    # print(model.head.__class__)
    x = torch.randn(52, 3, 384, 384)
    y = model(x)
    print(y.shape)  # should be [1, 1000]