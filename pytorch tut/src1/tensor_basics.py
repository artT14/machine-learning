import torch
import numpy as np

# TENSOR GENERATION
x_empty = torch.empty(3)
x_rand = torch.rand(2,2)
x_ones = torch.ones(2,2, dtype=torch.int)
x_custom = torch.tensor([2.5,0.1])

# TENSOR OPERATIONS
x = torch.rand(2,2)
y = torch.rand(2,2)
z = x + y
z = torch.add(x,y)
print(y)
y.add_(x) #modifes y
print(y)
y.sub_(x)
print(y)
y.mul_(x)
print(y)
y.div_(x)
print(y)

# TARGETING TENSOR VALUES
x = torch.rand(5,3)
print(x)
print(x[:,0]) #print only first column
print(x[1,:]) #print only second row
print(x[1,1].item()) #print item w/i tensor() specifier

# REARRANGING TENSORS
x = torch.rand(4,4)
print(x)
y = x.view(16)
print(y)
y = x.view(-1,8)
print(y)

# CONVERTING torch/numpty tensors
a = torch.ones(5)
print(a)
b = a.numpy()
print(type(b)) #NOTE: These point to same mem location on GPU be !!!careful!!!

a = np.ones(5)
print(a)
b = torch.from_numpy(a, dtype=torch.int)
print(b) #NOTE: These point to same mem location on GPU, !!!be careful!!!

# CPU to GPU communication
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5,device=device)
    x = torch.ones(5)
    y = y.to(device)
    z = x + y #THIS WILL BE PERFORMED ON THE GPU
    # z.numpy() #THIS throws error since z is on GPU
    z = z.to("cpu")
    z.numpy() # this will work
    
x = torch.ones(5, requires_grad=True)
