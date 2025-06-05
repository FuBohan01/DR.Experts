from kan import *
import time
# torch.set_default_dtype(torch.float64)

device = torch.device('cpu')
# print(device)
start = time.time()
# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[224,5,1], grid=3, k=3, seed=1, auto_save=False, device=device)

x = torch.randn(3, 224)
y = model(x)
print(y.shape)

end = time.time()
print(f"程序运行时间: {end - start:.4f} 秒")