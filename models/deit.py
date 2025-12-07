# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from . import open_clip
from .kan import *
from .din import DeepInterestNet
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
        return x  

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

    def forward(self, X1, X2, X1_hidden):
        """
        Forward pass of the Differential Attention module.
        
        Args:
        - X (Tensor): Input tensor.
        - λ (float): Scaling factor for the difference.
        
        Returns:
        - Tensor: Output tensor.
        """
        X1 = X1.unsqueeze(1)  # [bs, 1, 384]
        X1 = X1.expand(-1, X2.shape[1], -1)  # [bs, 577, 384]
        X = torch.cat([X1, X2], dim=-1)
        Q1 = self.W_q1(X1)
        K1 = self.W_k1(X1_hidden)
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
        result = (A1_softmax - self.alpha * A2_softmax) @ V
        # return result[:, 0] #[bs, 586, 384]
        return result #[bs, 576, 384]

# class KAN_Layer(nn.Module):
#     def __init__(self, dim, dim_in):
#         super().__init__()
#         self.f2 = nn.Linear(dim,dim)
#         # self.f2 = ConvBN(dim, dim, 1, with_bn=False)
#         self.f1 = nn.Linear(dim, dim)
#         self.g = KAN(width=[dim, dim], grid=5, k=3, seed=10, auto_save=False, device='cuda')
#         self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
#         self.act = nn.ReLU6()

#     def forward(self, diff, vit):# [bs, 577, 384]
        
#         x1, x2 = self.f1(diff), self.f2(vit)
#         x = x1 * x2
#         x = self.g(x)           
#         return x
    
# def get_dwconv(dim, kernel, bias):
#     return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

# class gnconv(nn.Module):
#     def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
#         super().__init__()
#         self.order = order
#         self.dims = [dim // 2 ** i for i in range(order)] # [24, 48, 96, 192, 384]
#         self.dims.reverse()
#         self.proj_in = nn.Conv2d(dim, 2*dim, 1)

#         if gflayer is None:
#             self.dwconv = get_dwconv(sum(self.dims), 7, True)
#         else:
#             self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        
#         self.proj_out = nn.Linear(dim, dim, 1)

#         self.pws = nn.ModuleList(
#             [nn.Linear(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
#         )

#         self.pwx = nn.ModuleList(
#             [nn.Linear(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
#         )

#         self.scale = s
#         self.proj_imgsize = nn.Conv1d(in_channels=576, out_channels=1, kernel_size=1)
#         self.proj_query = nn.Linear(self.dims[-1], self.dims[0])  # [384, 24]
#         # print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f'%self.scale)

    # def forward(self, x, img, mask=None, dummy=False):# x=[bs, 10, 384], img=[bs, 576, 384]
    #     B, N, C = img.shape #[B, 576, 384]
    #     H, W = 24, 24
    #     img = img.reshape(B, C, 24, 24).contiguous()

    #     fused_img = self.proj_in(img)
    #     pwa, abc = torch.split(fused_img, (self.dims[0], sum(self.dims)), dim=1) #[bs, 24, H, W] [bs, 744, H, W]

    #     dw_abc = self.dwconv(abc) * self.scale # [bs, 744, H, W]  
    #     dw_abc = dw_abc.reshape(B, N, -1).contiguous()  # [bs, 576, 744]
    #     dw_abc = self.proj_imgsize(dw_abc)  # [bs, 1, 744]
    #     dw_abc = dw_abc.expand(-1, 10, -1) # [bs, 10, 744]
    #     pwa = pwa.reshape(B, N, -1).contiguous()  # [bs, 576, 24]
    #     pwa = self.proj_imgsize(pwa)  # [bs, 1, 24]
    #     pwa = pwa.expand(-1, 10, -1) # [bs, 10, 24]

    #     dw_list = torch.split(dw_abc, self.dims, dim=-1) #[bs, 10, 24] [bs, 10, 48]...[bs, 10, 384]
    #     pw_q = self.proj_query(x) # [bs, 10, 24]
    #     cross = pw_q * dw_list[0] * pwa

    #     for i in range(self.order -1):
    #         pw_q = self.pwx[i](pw_q) # [bs, 10, 48]...[bs, 10, 384]
    #         cross = self.pws[i](cross) * dw_list[i+1] * pw_q

    #     x = self.proj_out(cross)

    #     return x
    
# class HorNet_Block(nn.Module):
#     r""" HorNet block
#     """
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
#         super().__init__()

#         self.norm1 = nn.LayerNorm(dim, eps=1e-6)
#         self.gnconv = gnconv(dim) # depthwise conv
#         self.norm2 = nn.LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)

#         self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
#                                     requires_grad=True) if layer_scale_init_value > 0 else None

#         self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
#                                     requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x, img):
#         B, N, C = x.shape
#         if self.gamma1 is not None:
#             gamma1 = self.gamma1.view(1,1,C)
#         else:
#             gamma1 = 1
#         x = x + self.drop_path(gamma1 * self.gnconv(x, self.norm1(img)))

#         input = x
#         x = self.norm2(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma2 is not None:
#             x = self.gamma2 * x

#         x = input + self.drop_path(x)
#         return x
    
# DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)
class dascore_vit_models(nn.Module):
    def __init__(self, img_size=384, model_size='small'):
        super().__init__()
        if model_size == 'small':
            self.backbone = vit_models(img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
                                norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block,num_classes=1)
        else:
            self.backbone = vit_models(img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, num_classes=1)
        
        if img_size == 384:
            self.N = 576
        else:
            self.N = 196
        
        if model_size == 'small':
            self.embed_dim = 384
        else:
            self.embed_dim = 768
        self.decoder = DeepInterestNet(self.embed_dim, 10)
        
        checkpoint = '/home/fubohan/Code/DIQA/checkpoint/daclip_ViT-B-32.pt'
        self.head, self.head_preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=checkpoint)
        degradations = ['motion-blurry','hazy','jpeg-compressed','low-light','noisy','raindrop','rainy','shadowed','snowy','uncompleted']

        self.L1 = nn.Linear(512, self.embed_dim)
        self.L2 = nn.Linear(self.N, 1)  # 196 is the number of patches in ViT-B-32
        self.diff_attention_list = nn.ModuleList([diff_attention(self.embed_dim) for _ in range(10)])
        # self.diff_attention = diff_attention(self.embed_dim)  # 384 is the embedding dimension of ViT-B-32

        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.text = tokenizer(degradations).cuda()
        self.lamda_init = nn.Parameter(torch.zeros(1, self.embed_dim), requires_grad=True)

        self.attn_norm = nn.LayerNorm(self.embed_dim)
        self.group_attn_ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.norm1 = nn.LayerNorm(512)
        self.L_hidden = nn.Linear(768, self.embed_dim)  # 

        self.score = nn.Linear(self.embed_dim, 1)

        self.norm_vit = nn.LayerNorm(self.embed_dim)  # 
        self.norm3_hidden_feature = nn.LayerNorm(self.embed_dim)  # 
        self.linears = nn.ModuleList([nn.Linear(512, self.embed_dim) for _ in range(10)])


    def mulithead(self, da_metrics, img_feature, hidden_feature):
        bs = da_metrics.shape[0]  # batch size
        group_attn = []
        for i in range(da_metrics.shape[1]):
            da_metrics_i = da_metrics[:, i, :]  # [bs, 384]
            img_feature = self.norm_vit(img_feature)  
            hidden_feature_i = hidden_feature[:, i, :, :]  # [bs, 577, 384]
            hidden_feature_i = self.norm3_hidden_feature(hidden_feature_i)  # [bs,
            diff_attn = self.diff_attention_list[i](da_metrics_i, img_feature, hidden_feature_i)  # [bs, 577, 384]

            group_attn.append(diff_attn)  
        # group_attn = torch.cat(group_attn, dim=-1)  # [bs, 577, 384 * nclass]
        group_attn = torch.stack(group_attn) 
        
        # Norm 和 FFN 
        group_attn = self.attn_norm(group_attn)  # [10, bs, 577, 384]
        # group_attn = self.attn_norm(group_attn)  
        group_attn = self.group_attn_ffn(group_attn) # [10, bs, 577, 16]
        # 
        group_attn = group_attn * (1 - self.lamda_init)
        return group_attn
        

    def forward(self, x):
        # pdb.set_trace()
        bs = x.shape[0]  # batch size
        img_feature = self.backbone(x) # [1, 577, 384]
 
        # da_img = 
        da_img = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=da_img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=da_img.device).view(1, 3, 1, 1)
        da_img = (da_img - mean) / std

        text_features = self.head.encode_text(self.text)
        da_img_feature, hiddens= self.head.encode_image(da_img, control= True) # shape=[1, 512] 
        da_img_feature = da_img_feature / da_img_feature.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 选择特定的分类类别 (例如 'jpeg-compressed', 'low-light' 对应索引 2, 3)
        selected_indices = [0,1,2,3,4,5,6,7,8,9]  # 根据degradations列表中的位置调整
        text_features_selected = text_features[selected_indices]  # [n_class, 512]

        # 先扩展维度
        text_features_expand = text_features_selected.unsqueeze(0).expand(da_img_feature.shape[0], -1, -1)  # [52, nclass, 512]
        da_img_feature_expand = da_img_feature.unsqueeze(1)  # [52, 1, 512]

        # 按元素相乘
        da_metrics = text_features_expand * da_img_feature_expand  # [bs, nclass, 512]
        da_metrics = self.norm1(da_metrics)  # [52, n_class, 512]
        # da_metrics = self.L1(da_metrics) # shape=[bs, n_class, 384]
        da_metrcis_list = []
        for i in range(len(selected_indices)):
            da_metrcis_list.append(self.linears[i](da_metrics[:, i, :]))  # [bs, 384]
        da_metrics = torch.stack(da_metrcis_list, dim=1)  # [bs, 10, 384]

        hidden_feature = hiddens[-1] 
        hidden_feature = hidden_feature.permute(1, 0, 2)  # [bs, 50, 768]
        hidden_feature = self.L_hidden(hidden_feature)  # [bs, 50, 384]
        hidden_feature = hidden_feature[:, 1:, :]         # [bs, 49, 384]
        hidden_feature = hidden_feature.reshape(bs, 7, 7, self.embed_dim)         # [bs, 7, 7, 384]
        hidden_feature = hidden_feature.permute(0, 3, 1, 2)            # [bs, 384, 7, 7]
        hidden_feature = F.interpolate(hidden_feature, size=(24, 24), mode='bilinear', align_corners=False)  # [bs, 384, 24, 24]
        hidden_feature = hidden_feature.permute(0, 2, 3, 1)            # [bs, 24, 24, 384]
        hidden_feature = hidden_feature.reshape(bs, -1, self.embed_dim)  # [bs, 576, 384]
        hidden_feature = hidden_feature.unsqueeze(1)  # [bs, 1, 576, 384]
        hidden_feature = hidden_feature.expand(-1, len(selected_indices), -1, -1)  # [bs, 10, 576, 384]
        da_token = da_metrics.unsqueeze(2)  # [bs, 10, 1, 384]
        hidden_feature = torch.cat((da_token, hidden_feature), dim=2)  # [bs,10, 577, 384]
        # img_feature = self.norm2(img_feature)  # [52, 576, 384]

        out = self.mulithead(da_metrics, img_feature, hidden_feature) # shape=[10, bs, 577, 384]

        out = out.permute(1, 0, 2, 3) #[bs, 10, 577, 384]


        score = self.decoder(out, img_feature)  # [bs, 577, 384]

        score = score[:, 0, :]

        score = self.score(score)  # [bs, 1]
        # score = score.view(bs, -1).sum(dim=1)
        # score = self.kan(score.squeeze())
        # score = self.L3(score.squeeze())
        # score = score.unsqueeze(1)
        return score

def build_deit_large(pretrained=False, img_size=384, model_size='small',pretrained_21k=True, infer=False, infer_model_path=None, **kwargs):
    # 创建主模型
    model = dascore_vit_models(img_size=img_size, model_size=model_size)
    # 如果需要加载backbone的预训练权重
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_' + model_size + '_' +  str(img_size) + '_'
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
    elif infer:
        assert infer_model_path != ""
        checkpoint = torch.load(
            infer_model_path, map_location="cpu", weights_only=False
        )
        state_dict = checkpoint["model"]
        model.load_state_dict(state_dict, strict=True)
        del checkpoint
        torch.cuda.empty_cache()
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
    model = build_deit_large(img_size=384,model_size='small')
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