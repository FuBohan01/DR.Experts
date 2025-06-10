import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
# from timm.models.registry import register_model

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=False):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # out = []
        x = self.stem(x)
        for stage in self.stages:
            res = x
            x = stage(x)
            # out.append(x)
            x = res * x
            # shape = x.shape

        # [2, 256, 7, 7]
        x = torch.flatten(self.avgpool(x), 1) 
        return self.head(x)



def starnet_s1(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], **kwargs)
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model



def starnet_s2(pretrained=False, **kwargs):
    model = StarNet(32, [1, 2, 6, 2], **kwargs)
    if pretrained:
        url = model_urls['starnet_s2']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model



def starnet_s3(pretrained=False, **kwargs):
    model = StarNet(32, [2, 2, 8, 4], **kwargs)
    if pretrained:
        url = model_urls['starnet_s3']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model



def starnet_s4(pretrained=False, **kwargs):
    model = StarNet(32, [3, 3, 12, 5], **kwargs)
    if pretrained:
        url = model_urls['starnet_s4']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model

class iqa(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, num_class = 1):
        super().__init__()
        self.backbone = StarNet(base_dim, depths, mlp_ratio, num_classes=num_class)
        self.head = nn.ModuleList()
        for i in range(len(depths)):
            embed_dim = base_dim * 2 ** i
            conv = ConvBN(embed_dim, 256)
            self.head.append(conv)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(256, 1)
    def forward(self, x):
        out = self.backbone(x)
        y = []
        for i in range(4):
            head_out = self.head[i](out[i])
            head_out = self.avgpool(head_out)
            head_out = self.linear(head_out)
            y.append(head_out)
        result = y[0]
        for tensor in y[1:]:
            result = result * tensor
        return result
    
def build_star(
    dims=32,
    depths=[3, 3, 12, 5],
    mlp_ratio = 4,
    num_classes = 1,
    pretrained=False,
    pretrained_model_path="",
    infer=False,
    infer_model_path="",
):
    model = StarNet(
        base_dim=dims,
        depths=depths,
        mlp_ratio=mlp_ratio,
        num_classes=num_classes
    )
    if pretrained:
        assert pretrained_model_path != ""
        checkpoint = torch.load(
            pretrained_model_path, map_location="cpu", weights_only=False
        )
        state_dict = checkpoint["state_dict"]
        del state_dict["head.weight"]
        del state_dict["head.bias"]
        del state_dict["norm.weight"]
        del state_dict["norm.bias"]
        del state_dict["norm.running_mean"]
        del state_dict["norm.running_var"]
        del state_dict["norm.num_batches_tracked"]
        model.load_state_dict(state_dict, strict=False)
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
    model = StarNet(32, [1, 2, 6, 2])
    print(model)
    # checkpoint = torch.load(
    #     "/home/fubohan/Code/DIQA-dev/checkpoint/starnet_s4.pth", map_location="cpu", weights_only=False
    # )
    # state_dict = checkpoint["state_dict"]
    # del state_dict["norm.weight"]
    # del state_dict["norm.bias"]
    # del state_dict["norm.running_mean"]
    # del state_dict["norm.running_var"]
    # del state_dict["norm.num_batches_tracked"]
    # del state_dict["head.weight"]
    # del state_dict["head.bias"]
    # model.backbone.load_state_dict(state_dict, strict=False)
    x = torch.randn(2, 3, 224, 224)
    from torchinfo import summary

    # summary(model, input_data=(x))
    print(model(x))