import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, trunc_normal_
from einops import rearrange, repeat
from typing import Callable, Optional, Union
# from visualizer import get_local
from torch import Tensor

try:
    from content_encoder import build_encoder
    from gabor import GaborLayer
except:
    from models.content_encoder import build_encoder
    from models.gabor import GaborLayer


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ConvolutionalGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        return x


class GaborResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GaborResidualBlock, self).__init__()
        self.conv1 = GaborLayer(
            in_channels, out_channels, kernel_size=3, padding=1, kernels=1, stride=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = ConvolutionalGLU(out_channels, None, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        out = F.relu(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = ConvolutionalGLU(out_channels, None, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        out = F.relu(out)
        return out


class BottomUpFPN(nn.Module):
    def __init__(self, in_channels_list, structural_enhance=False):
        """
        in_channels_list: e.g. [256, 512, 1024, 2048]
        """
        super(BottomUpFPN, self).__init__()
        assert len(in_channels_list) == 4

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, in_ch, kernel_size=1) for in_ch in in_channels_list]
        )

        self.down_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels_list[i],
                    in_channels_list[i + 1],
                    kernel_size=1,
                    stride=2,
                )
                for i in range(len(in_channels_list) - 1)
            ]
        )

        if structural_enhance:
            self.res_blocks = nn.ModuleList(
                [
                    GaborResidualBlock(in_channels_list[i + 1], in_channels_list[i + 1])
                    for i in range(len(in_channels_list) - 1)
                ]
            )

            self.res_block_c2 = GaborResidualBlock(
                in_channels_list[0], in_channels_list[0]
            )
        else:
            self.res_blocks = nn.ModuleList(
                [
                    ResidualBlock(in_channels_list[i + 1], in_channels_list[i + 1])
                    for i in range(len(in_channels_list) - 1)
                ]
            )

            self.res_block_c2 = ResidualBlock(in_channels_list[0], in_channels_list[0])

    def forward(self, features, return_all=False):
        C2, C3, C4, C5 = features

        # 1) lateral conv
        C2_lat = self.lateral_convs[0](C2)
        C3_lat = self.lateral_convs[1](C3)
        C4_lat = self.lateral_convs[2](C4)
        C5_lat = self.lateral_convs[3](C5)

        C2_lat = self.res_block_c2(C2_lat)

        x = self.down_convs[0](C2_lat)  # (B, C3_ch, H/8, W/8)
        x = x + C3_lat
        x = self.res_blocks[0](x)
        outC3 = x

        x = self.down_convs[1](x)  # (B, C4_ch, H/16, W/16)
        x = x + C4_lat
        x = self.res_blocks[1](x)
        outC4 = x

        x = self.down_convs[2](x)  # (B, C5_ch, H/32, W/32)
        x = x + C5_lat
        x = self.res_blocks[2](x)
        outC5 = x

        outC2 = C2_lat

        if return_all:
            return [outC2, outC3, outC4, outC5]
        else:
            return outC5


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        # self.norm = nn.BatchNorm2d(dim)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(
            dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio
        )
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.act(x)
        x = self.fc2(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_head=8, window=7):
        super().__init__()
        self.num_head = num_head
        self.window = window

        self.q = nn.Linear(dim, dim)
        self.q_cut = nn.Linear(dim, dim // 2)
        self.a = nn.Linear(dim, dim)
        self.l = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.e_conv = nn.Conv2d(dim // 2, dim // 2, 7, padding=3, groups=dim // 2)
        self.e_fore = nn.Linear(dim // 2, dim // 2)
        self.e_back = nn.Linear(dim // 2, dim // 2)

        self.proj = nn.Linear(dim // 2 * 3, dim)
        self.proj_e = nn.Linear(dim // 2 * 3, dim // 2)
        if window != 0:
            self.short_cut_linear = nn.Linear(dim // 2 * 3, dim // 2)  #####
            self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
            self.kv = nn.Linear(dim, dim)
            # self.m = nn.Parameter(torch.zeros(1, window, window, dim // 2), requires_grad=True)
            self.proj = nn.Linear(dim * 2, dim)
            self.proj_e = nn.Linear(dim * 2, dim // 2)

        self.act = nn.GELU()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm_e = LayerNorm(dim // 2, eps=1e-6, data_format="channels_last")

    def forward(self, x, x_e):
        B, H, W, C = x.size()
        x = self.norm(x)
        x_e = self.norm_e(x_e)
        if self.window != 0:
            short_cut = torch.cat([x, x_e], dim=3)  ##########
            short_cut = short_cut.permute(0, 3, 1, 2).contiguous()  #############

        q = self.q(x)
        cutted_x = self.q_cut(x)
        x = self.l(x).permute(0, 3, 1, 2).contiguous()
        x = self.act(x)

        a = self.conv(x)
        a = a.permute(0, 2, 3, 1).contiguous()
        a = self.a(a)

        if self.window != 0:
            b = x.permute(0, 2, 3, 1).contiguous()
            kv = self.kv(b)
            kv = (
                kv.reshape(B, H * W, 2, self.num_head, C // self.num_head // 2)
                .permute(2, 0, 3, 1, 4)
                .contiguous()
            )
            k, v = kv.unbind(0)
            ####
            short_cut = self.pool(short_cut).permute(0, 2, 3, 1).contiguous()
            short_cut = self.short_cut_linear(
                short_cut
            )  # (B,7,7,3DIM//2)->(B,7,7,DIM//2)
            ####
            short_cut = (
                short_cut.reshape(B, -1, self.num_head, C // self.num_head // 2)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            # m = self.m.reshape(1, -1, self.num_head, C // self.num_head // 2).permute(0, 2, 1, 3).expand(B, -1, -1, -1)
            # print(m.shape,short_cut.shape)
            m = short_cut
            attn = (m * (C // self.num_head // 2) ** -0.5) @ k.transpose(
                -2, -1
            ).contiguous()
            attn = attn.softmax(dim=-1)
            attn = (
                (attn @ v)
                .reshape(
                    B, self.num_head, self.window, self.window, C // self.num_head // 2
                )
                .permute(0, 1, 4, 2, 3)
                .contiguous()
                .reshape(B, C // 2, self.window, self.window)
            )
            attn = (
                F.interpolate(attn, (H, W), mode="bilinear", align_corners=False)
                .permute(0, 2, 3, 1)
                .contiguous()
            )

        x_e = self.e_back(
            self.e_conv(self.e_fore(x_e).permute(0, 3, 1, 2).contiguous())
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        cutted_x = cutted_x * x_e
        x = q * a

        if self.window != 0:
            x = torch.cat([x, attn, cutted_x], dim=3)
        else:
            x = torch.cat([x, cutted_x], dim=3)

        x_e = self.proj_e(x)
        x = self.proj(x)

        return x, x_e


class Block(nn.Module):
    def __init__(
        self,
        index,
        dim,
        num_head,
        window=7,
        mlp_ratio=4.0,
        drop_path=0.0,
        block_index=0,
        last_block_index=50,
    ):
        super().__init__()

        self.index = index
        layer_scale_init_value = 1e-6
        if block_index > last_block_index:
            window = 0
        self.attn = Attention(dim, num_head, window=window)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = MLP(dim, mlp_ratio)
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )
        self.layer_scale_1_e = nn.Parameter(
            layer_scale_init_value * torch.ones((dim // 2)), requires_grad=True
        )
        self.layer_scale_2_e = nn.Parameter(
            layer_scale_init_value * torch.ones((dim // 2)), requires_grad=True
        )
        # self.mlp_e1 = MLP(dim//2, mlp_ratio)
        self.mlp_e2 = MLP(dim // 2, mlp_ratio)

    def forward(self, x, x_e):
        res_x, res_e = x, x_e
        x, x_e = self.attn(x, x_e)

        x_e = res_e + self.drop_path(
            self.layer_scale_1_e.unsqueeze(0).unsqueeze(0) * x_e
        )
        x = res_x + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * x)

        x_e = x_e + self.drop_path(
            self.layer_scale_2_e.unsqueeze(0).unsqueeze(0) * self.mlp_e2(x_e)
        )
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(x)
        )
        return x, x_e


class DFormer_model(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        windows=[7, 7, 7, 7],
        mlp_ratios=[4, 4, 4, 4],
        last_block=[50, 50, 50, 50],
        num_heads=[2, 4, 10, 16],
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        drop_path_rate=0.0,
        drop_rate=0.0,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
        )

        self.downsample_layers_e = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem_e = nn.Sequential(
            nn.Conv2d(1, dims[0] // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 4),
            nn.GELU(),
            nn.Conv2d(dims[0] // 4, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
        )

        self.downsample_layers.append(stem)
        self.downsample_layers_e.append(stem_e)

        for i in range(len(dims) - 1):
            stride = 2
            downsample_layer = nn.Sequential(
                nn.BatchNorm2d(dims[i]),
                nn.Conv2d(
                    dims[i], dims[i + 1], kernel_size=3, stride=stride, padding=1
                ),
            )
            self.downsample_layers.append(downsample_layer)

            downsample_layer_e = nn.Sequential(
                nn.BatchNorm2d(dims[i] // 2),
                nn.Conv2d(
                    dims[i] // 2,
                    dims[i + 1] // 2,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                ),
            )
            self.downsample_layers_e.append(downsample_layer_e)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[
                    Block(
                        index=cur + j,
                        dim=dims[i],
                        window=windows[i],
                        drop_path=dp_rates[cur + j],
                        num_head=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        block_index=depths[i] - j,
                        last_block_index=last_block[i],
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            # nn.init.constant_(m.bias, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, x_e, x_o):
        x_e = x_e.unsqueeze(1)
        assert len(x_e.shape) == 4
        out_x, out_xe, out = [], [], []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x_e = self.downsample_layers_e[i](x_e)
            x = x.permute(0, 2, 3, 1).contiguous()
            x_e = x_e.permute(0, 2, 3, 1).contiguous()
            for blk in self.stages[i]:
                x, x_e = blk(x, x_e)
            x = x.permute(0, 3, 1, 2).contiguous()
            x_e = x_e.permute(0, 3, 1, 2).contiguous()
            out.append(torch.cat([x, x_e], dim=1))
            out_x.append(x)
            out_xe.append(x_e)
        x = torch.cat([x, x_e], dim=1)
        return (
            x.mean([-2, -1]).unsqueeze(
                1
            ),  # global average pooling, (N, C, H, W) -> (N, C)
            rearrange(x, "b c h w -> b (h w) c"),
            out,
            out_x,
            out_xe,
        )

    def forward(self, x, x_e, x_o):
        x, ref, out, out_x, out_xe = self.forward_features(x, x_e, x_o)
        return x, ref, out, out_x, out_xe


class ModifiedTransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )
        self.norm3 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    def _get_activation_fn(self, activation: str) -> Callable[[Tensor], Tensor]:
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._mha_block(
                self.norm1(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._sa_block(
                self.norm2(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm2(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # @get_local("map")
    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x, map = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=True,
        )
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class Experts_MOS(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        juery_nums=6,
    ):
        super().__init__()
        self.juery = juery_nums * 2

        orders_layer = ModifiedTransformerDecoderLayer(
            d_model=embed_dim,
            dropout=0.0,
            nhead=6,
            activation=F.gelu,
            batch_first=True,
            dim_feedforward=(embed_dim * 4),
            norm_first=True,
        )
        self.orders_decoder = nn.TransformerDecoder(orders_layer, num_layers=1)

        self.bunch_embedding = nn.Parameter(torch.randn(1, self.juery, embed_dim))
        self.heads = nn.Sequential(nn.Linear(embed_dim, 1, bias=False))
        trunc_normal_(self.bunch_embedding, std=0.02)

        self.embed_dim = embed_dim

        self.append_mask = torch.zeros((6, self.juery // 2, 49)).bool().cuda()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            # nn.init.constant_(m.bias, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def mask_convert(self, x_o):
        for i in range(len(x_o)):
            x_o[i] = torch.cat(
                [
                    F.interpolate(
                        x_o[i].unsqueeze(0),
                        (7, 7),
                        mode="bilinear",
                        align_corners=True,
                    ).view(x_o[i].shape[0], -1, 1)
                    > 0.5,
                ],
                dim=1,
            )
        return x_o

    def _get_attn_mask(self, attn_mask):
        count_mask = attn_mask
        B = count_mask.shape[0]
        inv_mask = attn_mask
        attn_mask = ~attn_mask
        attn_mask = repeat(attn_mask, "b k 1 -> (b h) k 1", h=6)
        attn_mask = repeat(attn_mask, "n k 1 -> n (q k) 1", q=self.juery // 2).squeeze(
            -1
        )
        attn_mask = rearrange(attn_mask, "n (q k) -> n q k", q=self.juery // 2)
        attn_mask = torch.cat([attn_mask, self.append_mask], dim=2)

        inv_mask = repeat(inv_mask, "b k 1 -> (b h) k 1", h=6)
        inv_mask = repeat(inv_mask, "n k 1 -> n (q k) 1", q=self.juery // 2).squeeze(-1)
        inv_mask = rearrange(inv_mask, "n (q k) -> n q k", q=self.juery // 2)
        inv_mask = torch.cat([inv_mask, self.append_mask], dim=2)
        attn_mask = torch.cat([attn_mask, inv_mask], dim=1)

        true_count = count_mask.float().sum(dim=1).expand(B, self.juery // 2)
        false_count = 49 - true_count
        count = torch.cat([true_count, false_count], dim=1) / 49.0
        return attn_mask, count

    def forward(self, x, x_o, ref, ref_o, ref_oe, c):

        B = x.shape[0]
        img_feats = ref

        bunch_embedding = self.bunch_embedding.expand(B, -1, -1)

        x = bunch_embedding * x.expand(B, self.juery, -1)

        ref_oe = rearrange(ref_oe, "b c h w -> b (h w) c")
        ref_oe = repeat(ref_oe, "b n c -> b n (c 3)")

        results = []

        x_o = self.mask_convert(x_o)

        for i in range(B):
            cur_res = []
            cur_orders_map = x_o[i]
            cur_img_feats = img_feats[i].unsqueeze(0)
            cur_ref_oe = ref_oe[i].unsqueeze(0)
            cur_ref = torch.cat([cur_img_feats, cur_ref_oe], dim=1)
            cur_query = x[i].unsqueeze(0)
            for j in range(cur_orders_map.shape[0]):
                cur_lay_map, count = self._get_attn_mask(
                    cur_orders_map[j, :, :].unsqueeze(0)
                )
                order_q = self.orders_decoder(
                    cur_query, cur_ref, memory_mask=cur_lay_map
                )
                order_q = self.heads(order_q)
                print(order_q)
                order_q = order_q.view(1, -1).mean(dim=1, keepdim=True)
                cur_res.append(order_q)
            results.append(torch.cat(cur_res, dim=1).mean(dim=1))
            pass

        x = torch.cat(results, dim=0)

        x = x.view(B, -1)

        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DIQA(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        windows=[7, 7, 7, 7],
        mlp_ratios=[4, 4, 4, 4],
        last_block=[50, 50, 50, 50],
        num_heads=[2, 4, 10, 16],
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        drop_path_rate=0.0,
        drop_rate=0.0,
        **kwargs,
    ):
        super().__init__()
        self.backbone = DFormer_model(
            img_size=img_size,
            in_chans=in_chans,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            windows=windows,
            mlp_ratios=mlp_ratios,
            last_block=last_block,
            num_heads=num_heads,
            layer_scale_init_value=layer_scale_init_value,
            head_init_scale=head_init_scale,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )
        self.content_backbone = build_encoder(
            img_size=224,
            pretrained=True,
            pretrained_model_path="checkpoints/deit_3_small_224_21k.pth",
        )
        self.target_dim = dims[-1] // 2 * 3
        self.content_convert = nn.Sequential(
            nn.Linear(384, self.target_dim),
            nn.Sigmoid(),
        )
        # self.mutual_learn = nn.Sequential(
        self.mutual_learn = nn.Sequential(
            nn.Linear(self.target_dim * 2, self.target_dim),
            nn.Sigmoid(),
        )
        self.mutual_learn_s2 = nn.Sequential(
            nn.Linear(self.target_dim * 1, self.target_dim),
            nn.Sigmoid(),
        )
        self.low_level_aware = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        self.low_level_learn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dims[0] // 2 * 3, dims[0] // 2),
                    nn.Sigmoid(),
                ),
                nn.Sequential(
                    nn.Linear(dims[1] // 2 * 3, dims[1] // 2),
                    nn.Sigmoid(),
                ),
                nn.Sequential(
                    nn.Linear(dims[2] // 2 * 3, dims[2] // 2),
                    nn.Sigmoid(),
                ),
                nn.Sequential(
                    nn.Linear(dims[3] // 2 * 3, dims[3] // 2),
                    nn.Sigmoid(),
                ),
            ]
        )
        self.low_level_convert = nn.Sequential(
            nn.Linear(sum(dims) // 2, self.target_dim),
            nn.Sigmoid(),
        )
        self.gabor_fpn = BottomUpFPN(
            in_channels_list=[dims[0] // 2, dims[1] // 2, dims[2] // 2, dims[3] // 2],
            structural_enhance=False,
        )
        self.rgb_fpn = BottomUpFPN(
            in_channels_list=[
                dims[0],
                dims[1],
                dims[2],
                dims[3],
            ],
            structural_enhance=False,
        )
        self.pred = Experts_MOS(dims[-1] // 2 * 3, juery_nums=6)

    def forward(self, x, x_e, x_o):
        c = self.content_backbone(F.interpolate(x, (224, 224), mode="bilinear"))
        _, ref, out, out_x, out_xe = self.backbone(x, x_e, x_o)
        out_xe = self.gabor_fpn(out_xe)
        out_x = self.rgb_fpn(out_x)
        for i in range(len(out)):
            out[i] = self.low_level_aware(out[i])
            out[i] = self.low_level_learn[i](out[i])
        out = torch.cat(out, dim=1)
        l_out = self.low_level_convert(out).unsqueeze(1)
        c = self.content_convert(c).unsqueeze(1)
        x = (
            self.mutual_learn(
                torch.cat(
                    [
                        l_out,
                        c,
                    ],
                    dim=2,
                )
            )
            + l_out
            # + c
        )
        x = self.mutual_learn_s2(x) + l_out
        x = self.pred(x, x_o, ref, out_x, out_xe, x)
        return x


def build_dformer(
    dims=[32, 64, 128, 256],
    mlp_ratios=[8, 8, 4, 4],
    depths=[3, 3, 5, 2],
    num_heads=[1, 2, 4, 8],
    windows=[0, 7, 7, 7],
    pretrained=False,
    pretrained_model_path="",
    infer=False,
    infer_model_path="",
):
    model = DIQA(
        dims=dims,
        mlp_ratios=mlp_ratios,
        depths=depths,
        num_heads=num_heads,
        windows=windows,
    )
    if pretrained:
        assert pretrained_model_path != ""
        checkpoint = torch.load(
            pretrained_model_path, map_location="cpu", weights_only=False
        )
        state_dict = checkpoint["state_dict"]
        del state_dict["pred.weight"]
        del state_dict["pred.bias"]
        model.backbone.load_state_dict(state_dict, strict=True)
        # for param in model.backbone.parameters():
        #     param.requires_grad = False
        del checkpoint
        torch.cuda.empty_cache()
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


if __name__ == "__main__":
    model = DIQA(
        dims=[64, 128, 256, 512],
        mlp_ratios=[8, 8, 4, 4],
        depths=[2, 2, 4, 2],
        num_heads=[1, 2, 4, 8],
        windows=[0, 7, 7, 7],
    )
    print(model)
    checkpoint = torch.load(
        "checkpoints/DFormer_Small.pth", map_location="cpu", weights_only=False
    )
    state_dict = checkpoint["state_dict"]
    del state_dict["pred.weight"]
    del state_dict["pred.bias"]
    model.backbone.load_state_dict(state_dict, strict=True)
    x = torch.randn(2, 3, 224, 224)
    x_e = torch.randn(2, 224, 224)
    x_o = [torch.randn(6, 224, 224), torch.randn(5, 224, 224)]
    from torchinfo import summary

    summary(model, input_data=(x, x_e, x_o))
