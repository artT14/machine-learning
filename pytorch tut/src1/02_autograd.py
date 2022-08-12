import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2

print(y)

z = y*y*2

print(z)

w = z.mean()
print(w)

# w.backward() #dw/dx
# print(x.grad)

v = torch.tensor([0.1,1.0,0.001], dtype=torch.float32)
z.backward(v) #dz/dx
print(x.grad)

# REMOVING AUTO GRADIENT
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():