from kan import *
import time
from starnet import ConvBN
# torch.set_default_dtype(torch.float64)

class Block(nn.Module):
    def __init__(self, dim, dim_in):
        super().__init__()
        self.dim_reduce = ConvBN(dim_in, dim, 1, with_bn=False)
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, dim, 1, with_bn=False)
        # self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.g = KAN(width=[dim,dim, dim], grid=5, k=3, seed=1, auto_save=False, device='cpu')
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()

    def forward(self, x):
        x = self.dim_reduce(x)  # Reduce dimension first
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        bs, dim, h, w = x.shape
        x = x.permute(0, 2, 3, 1)       # [3, 14, 14, 32]
        x = x.reshape(bs*h*w, dim)     # [3* 196, 32]
        x = self.g(x)                 # [3*196, 32]
        x = x.reshape(bs, h*w, dim)  # [3, 196, 32]
        x = x.permute(0, 2, 1)       # [3, 32, 196]
        x = x.reshape(bs, dim, h, w)    # [3, 32, 14, 14]
        x = self.dwconv2(x)
        x = input + x
        return x

device = torch.device('cpu')
# print(device)
start = time.time()
# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
# model = KAN(width=[384, 1], grid=5, k=3, seed=1, auto_save=False, device=device)

x = torch.randn(3, 197 ,384)
x1 = x[:, 0, :]
x2 = x[:, 1:, :]
x = x1.unsqueeze(1) + x2 # [3, 1, 384] + [3, 196, 384] -> [3, 196, 384]
x = x.permute(0, 2, 1)        # [3, 384, 197]
x = x.view(3, 384, 14, 14)    # [3, 384, 14, 14]
block = Block(32, 384)
y = block(x)
print(y.shape)
# y = model(x)
# print(y.shape)  # 输出形状应为 [3, 128, 197]

end = time.time()
print(f"程序运行时间: {end - start:.4f} 秒")