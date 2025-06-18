import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.fft

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)


class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)] # [24, 48, 96, 192, 384]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        
        self.proj_out = nn.Linear(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Linear(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )

        self.scale = s
        self.proj_imgsize = nn.Conv1d(in_channels=576, out_channels=1, kernel_size=1)
        self.proj_query = nn.Linear(self.dims[-1], self.dims[0])  # [384, 24]
        # print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f'%self.scale)

    def forward(self, x, img, mask=None, dummy=False):# x=[bs, 10, 384], img=[bs, 576, 384]
        B, N, C = img.shape #[B, 576, 384]
        H, W = 24, 24
        img = img.reshape(B, C, 24, 24).contiguous()

        fused_img = self.proj_in(img)
        _, abc = torch.split(fused_img, (self.dims[0], sum(self.dims)), dim=1) #[bs, 24, H, W] [bs, 744, H, W]

        dw_abc = self.dwconv(abc) * self.scale # [bs, 744, H, W]  
        dw_abc = dw_abc.reshape(B, N, -1).contiguous()  # [bs, 576, 744]
        dw_abc = self.proj_imgsize(dw_abc)  # [bs, 1, 744]
        dw_abc = dw_abc.expand(-1, 10, -1) # [bs, 10, 744]

        dw_list = torch.split(dw_abc, self.dims, dim=-1) #[bs, 10, 24] [bs, 10, 48]...[bs, 10, 384]
        pwa = self.proj_query(x) # [bs, 10, 24]
        cross = pwa * dw_list[0]

        for i in range(self.order -1):
            cross = self.pws[i](cross) * dw_list[i+1]

        x = self.proj_out(cross)

        return x
class Block(nn.Module):
    r""" HorNet block
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.gnconv = gnconv(dim) # depthwise conv
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, img):
        B, N, C = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(1,1,C)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(x, self.norm1(img)))

        input = x
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
if __name__ == '__main__':
    x = torch.randn(2, 10, 384)
    img = torch.randn(2, 576, 384)
    model = Block(384)
    out = model(x, img)
    print(out.shape)  # Should be [2, 64, 14, 8]